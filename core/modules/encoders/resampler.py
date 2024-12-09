import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


class ImageProjModel(nn.Module):
    """Projection Model"""

    def __init__(
        self,
        cross_attention_dim=1024,
        clip_embeddings_dim=1024,
        clip_extra_context_tokens=4,
    ):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * cross_attention_dim
        )
        self.norm = nn.LayerNorm(cross_attention_dim)

    def forward(self, image_embeds):
        # embeds = image_embeds
        embeds = image_embeds.type(list(self.proj.parameters())[0].dtype)
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.cross_attention_dim
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens


# FFN
def FeedForward(dim, mult=4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False),
    )


def reshape_tensor(x, heads):
    bs, length, width = x.shape
    # (bs, length, width) --> (bs, length, n_heads, dim_per_head)
    x = x.view(bs, length, heads, -1)
    # (bs, length, n_heads, dim_per_head) --> (bs, n_heads, length, dim_per_head)
    x = x.transpose(1, 2)
    # (bs, n_heads, length, dim_per_head) --> (bs*n_heads, length, dim_per_head)
    x = x.reshape(bs, heads, length, -1)
    return x


class PerceiverAttention(nn.Module):
    def __init__(self, *, dim, dim_head=64, heads=8):
        super().__init__()
        self.scale = dim_head**-0.5
        self.dim_head = dim_head
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, latents):
        """
        Args:
            x (torch.Tensor): image features
                shape (b, n1, D)
            latent (torch.Tensor): latent features
                shape (b, n2, D)
        """
        x = self.norm1(x)
        latents = self.norm2(latents)

        b, l, _ = latents.shape

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        q = reshape_tensor(q, self.heads)
        k = reshape_tensor(k, self.heads)
        v = reshape_tensor(v, self.heads)

        # attention
        scale = 1 / math.sqrt(math.sqrt(self.dim_head))
        # More stable with f16 than dividing afterwards
        weight = (q * scale) @ (k * scale).transpose(-2, -1)
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        out = weight @ v

        out = out.permute(0, 2, 1, 3).reshape(b, l, -1)

        return self.to_out(out)


class Resampler(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        video_length=None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.video_length = video_length
        if video_length is not None:
            num_queries = num_queries * video_length
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) / dim**0.5)
        self.proj_in = nn.Linear(embedding_dim, dim)
        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x):
        latents = self.latents.repeat(x.size(0), 1, 1)  # B (T L) C
        x = self.proj_in(x)

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        latents = self.proj_out(latents)
        latents = self.norm_out(latents)  # B L C or B (T L) C

        return latents


class CameraPoseQueryTransformer(nn.Module):
    def __init__(
        self,
        dim=1024,
        depth=8,
        dim_head=64,
        heads=16,
        num_queries=8,
        embedding_dim=768,
        output_dim=1024,
        ff_mult=4,
        num_views=None,
        use_multi_view_attention=True,
    ):
        super().__init__()

        self.num_queries = num_queries
        self.num_views = num_views
        assert num_views is not None, "video_length must be given."
        self.use_multi_view_attention = use_multi_view_attention
        self.camera_pose_embedding_layers = nn.Sequential(
            nn.Linear(12, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        nn.init.zeros_(self.camera_pose_embedding_layers[-1].weight)
        nn.init.zeros_(self.camera_pose_embedding_layers[-1].bias)

        self.latents = nn.Parameter(
            torch.randn(1, num_views * num_queries, dim) / dim**0.5
        )

        self.proj_in = nn.Linear(embedding_dim, dim)

        self.proj_out = nn.Linear(dim, output_dim)
        self.norm_out = nn.LayerNorm(output_dim)

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                        FeedForward(dim=dim, mult=ff_mult),
                    ]
                )
            )

    def forward(self, x, camera_poses):
        # camera_poses: (b, t, 12)
        batch_size, num_views, _ = camera_poses.shape
        # latents: (1, t*q, d) -> (b, t*q, d)
        latents = self.latents.repeat(batch_size, 1, 1)
        x = self.proj_in(x)
        # camera_poses: (b*t, 12)
        camera_poses = rearrange(camera_poses, "b t d -> (b t) d", t=num_views)
        camera_poses = self.camera_pose_embedding_layers(
            camera_poses
        )  # camera_poses: (b*t, d)
        # camera_poses: (b, t, d)
        camera_poses = rearrange(camera_poses, "(b t) d -> b t d", t=num_views)
        # camera_poses: (b, t*q, d)
        camera_poses = repeat(camera_poses, "b t d -> b (t q) d", q=self.num_queries)

        latents = latents + camera_poses  # b, t*q, d

        latents = rearrange(
            latents,
            "b (t q) d -> (b t) q d",
            b=batch_size,
            t=num_views,
            q=self.num_queries,
        )  # (b*t, q, d)

        _, x_seq_size, _ = x.shape
        for layer_idx, (attn, ff) in enumerate(self.layers):
            if self.use_multi_view_attention and layer_idx % 2 == 1:
                # latents: (b*t, q, d)
                latents = rearrange(
                    latents,
                    "(b t) q d -> b (t q) d",
                    b=batch_size,
                    t=num_views,
                    q=self.num_queries,
                )
                # x: (b*t, s, d)
                x = rearrange(
                    x, "(b t) s d -> b (t s) d", b=batch_size, t=num_views, s=x_seq_size
                )

                # print("After rearrange: latents.shape=", latents.shape)
                # print("After rearrange: x.shape=", camera_poses.shape)
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents
            if self.use_multi_view_attention and layer_idx % 2 == 1:
                # latents: (b*q, t, d)
                latents = rearrange(
                    latents,
                    "b (t q) d -> (b t) q d",
                    b=batch_size,
                    t=num_views,
                    q=self.num_queries,
                )
                # x: (b*s, t, d)
                x = rearrange(
                    x, "b (t s) d -> (b t) s d", b=batch_size, t=num_views, s=x_seq_size
                )
        latents = self.proj_out(latents)
        latents = self.norm_out(latents)  # B L C or B (T L) C
        return latents
