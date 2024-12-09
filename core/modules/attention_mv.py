import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from core.common import gradient_checkpoint

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False

print(f"XFORMERS_IS_AVAILBLE: {XFORMERS_IS_AVAILBLE}")


def get_group_norm_layer(in_channels):
    if in_channels < 32:
        if in_channels % 2 == 0:
            num_groups = in_channels // 2
        else:
            num_groups = in_channels
    else:
        num_groups = 32
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")


class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.0):
        super().__init__()
        inner_dim = int(dim * mult)
        if dim_out is None:
            dim_out = dim
        project_in = (
            nn.Sequential(nn.Linear(dim, inner_dim), nn.GELU())
            if not glu
            else GEGLU(dim, inner_dim)
        )

        self.net = nn.Sequential(
            project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


class SpatialTemporalAttention(nn.Module):
    """Uses xformers to implement efficient epipolar masking for cross-attention between views."""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        if context_dim is None:
            context_dim = query_dim

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), nn.Dropout(dropout)
        )
        self.attention_op = None

    def forward(self, x, context=None, enhance_multi_view_correspondence=False):
        q = self.to_q(x)
        if context is None:
            context = x
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        if enhance_multi_view_correspondence:
            with torch.no_grad():
                normalized_x = torch.nn.functional.normalize(x.detach(), p=2, dim=-1)
                cosine_sim_map = torch.bmm(normalized_x, normalized_x.transpose(-1, -2))
                attn_bias = torch.where(cosine_sim_map > 0.0, 0.0, -1e9).to(
                    dtype=q.dtype
                )
                attn_bias = rearrange(
                    attn_bias.unsqueeze(1).expand(-1, self.heads, -1, -1),
                    "b h d1 d2 -> (b h) d1 d2",
                ).detach()
        else:
            attn_bias = None

        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=attn_bias, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        del q, k, v, attn_bias
        return self.to_out(out)


class MultiViewSelfAttentionTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        n_heads,
        d_head,
        dropout=0.0,
        gated_ff=True,
        use_checkpoint=True,
        full_spatial_temporal_attention=False,
        enhance_multi_view_correspondence=False,
    ):
        super().__init__()
        attn_cls = SpatialTemporalAttention
        # self.self_attention_only = self_attention_only
        self.attn1 = attn_cls(
            query_dim=dim,
            heads=n_heads,
            dim_head=d_head,
            dropout=dropout,
            context_dim=None,
        )  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)

        if enhance_multi_view_correspondence:
            # Zero initalization when MVCorr is enabled.
            zero_module_fn = zero_module
        else:

            def zero_module_fn(x):
                return x

        self.attn2 = zero_module_fn(
            attn_cls(
                query_dim=dim,
                heads=n_heads,
                dim_head=d_head,
                dropout=dropout,
                context_dim=None,
            )
        )  # is self-attn if context is none

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.use_checkpoint = use_checkpoint
        self.full_spatial_temporal_attention = full_spatial_temporal_attention
        self.enhance_multi_view_correspondence = enhance_multi_view_correspondence

    def forward(self, x, time_steps=None):
        return gradient_checkpoint(
            self.many_stream_forward, (x, time_steps), None, flag=self.use_checkpoint
        )

    def many_stream_forward(self, x, time_steps=None):
        n, v, hw = x.shape[:3]
        x = rearrange(x, "n v hw c -> n (v hw) c")
        x = (
            self.attn1(
                self.norm1(x), context=None, enhance_multi_view_correspondence=False
            )
            + x
        )
        if not self.full_spatial_temporal_attention:
            x = rearrange(x, "n (v hw) c -> n v hw c", v=v)
            x = rearrange(x, "n v hw c -> (n v) hw c")
        x = (
            self.attn2(
                self.norm2(x),
                context=None,
                enhance_multi_view_correspondence=self.enhance_multi_view_correspondence
                and hw <= 256,
            )
            + x
        )
        x = self.ff(self.norm3(x)) + x
        if self.full_spatial_temporal_attention:
            x = rearrange(x, "n (v hw) c -> n v hw c", v=v)
        else:
            x = rearrange(x, "(n v) hw c -> n v hw c", v=v)
        return x


class MultiViewSelfAttentionTransformer(nn.Module):
    """Spatial Transformer block with post init to add cross attn."""

    def __init__(
        self,
        in_channels,
        n_heads,
        d_head,
        num_views,
        depth=1,
        dropout=0.0,
        use_linear=True,
        use_checkpoint=True,
        zero_out_initialization=True,
        full_spatial_temporal_attention=False,
        enhance_multi_view_correspondence=False,
    ):
        super().__init__()
        self.num_views = num_views
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = get_group_norm_layer(in_channels)
        if not use_linear:
            self.proj_in = nn.Conv2d(
                in_channels, inner_dim, kernel_size=1, stride=1, padding=0
            )
        else:
            self.proj_in = nn.Linear(in_channels, inner_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                MultiViewSelfAttentionTransformerBlock(
                    inner_dim,
                    n_heads,
                    d_head,
                    dropout=dropout,
                    use_checkpoint=use_checkpoint,
                    full_spatial_temporal_attention=full_spatial_temporal_attention,
                    enhance_multi_view_correspondence=enhance_multi_view_correspondence,
                )
                for d in range(depth)
            ]
        )
        self.zero_out_initialization = zero_out_initialization

        if zero_out_initialization:
            _zero_func = zero_module
        else:

            def _zero_func(x):
                return x

        if not use_linear:
            self.proj_out = _zero_func(
                nn.Conv2d(inner_dim, in_channels, kernel_size=1, stride=1, padding=0)
            )
        else:
            self.proj_out = _zero_func(nn.Linear(inner_dim, in_channels))

        self.use_linear = use_linear

    def forward(self, x, time_steps=None):
        # x: bt c h w
        _, c, h, w = x.shape
        n_views = self.num_views
        x_in = x
        x = self.norm(x)
        x = rearrange(x, "(n v) c h w -> n v (h w) c", v=n_views)

        if self.use_linear:
            x = rearrange(x, "n v x c -> (n v) x c")
            x = self.proj_in(x)
            x = rearrange(x, "(n v) x c -> n v x c", v=n_views)
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, time_steps=time_steps)
        if self.use_linear:
            x = rearrange(x, "n v x c -> (n v) x c")
            x = self.proj_out(x)
            x = rearrange(x, "(n v) x c -> n v x c", v=n_views)

        x = rearrange(x, "n v (h w) c -> (n v) c h w", h=h, w=w).contiguous()

        return x + x_in
