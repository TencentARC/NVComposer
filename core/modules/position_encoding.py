# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, token_tensors):
        # input: (B,C,H,W)
        x = token_tensors
        h, w = x.shape[-2:]
        identity_map = torch.ones((h, w), device=x.device)
        y_embed = identity_map.cumsum(0, dtype=torch.float32)
        x_embed = identity_map.cumsum(1, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[-1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
        ).flatten(2)
        pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1)
        batch_pos = pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return batch_pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, n_pos_x=16, n_pos_y=16, num_pos_feats=64):
        super().__init__()
        self.row_embed = nn.Embedding(n_pos_y, num_pos_feats)
        self.col_embed = nn.Embedding(n_pos_x, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, token_tensors):
        # input: (B,C,H,W)
        x = token_tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        ).permute(2, 0, 1)
        batch_pos = pos.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return batch_pos


def build_position_encoding(num_pos_feats=64, n_pos_x=16, n_pos_y=16, is_learned=False):
    if is_learned:
        position_embedding = PositionEmbeddingLearned(n_pos_x, n_pos_y, num_pos_feats)
    else:
        position_embedding = PositionEmbeddingSine(num_pos_feats, normalize=True)

    return position_embedding
