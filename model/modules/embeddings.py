import math

import torch
from torch import nn


class Identity(nn.Module):
    def forward(self, x, *args):
        return x


# too lazy to use it. would need to retrain.
class AdaptiveGroupNormalization(nn.Module):
    def __init__(self, input_channels, timestep_channels, embedding_dim):
        super(AdaptiveGroupNormalization, self).__init__()
        self.z_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, input_channels),
            nn.SiLU(),
            nn.Linear(input_channels, input_channels),
        )
        self.t_mlp = nn.Sequential(nn.Linear(timestep_channels, input_channels * 2), nn.SiLU())

    def forward(self, x, t, embeddings):
        embeddings = self.embeddings_mlp(embeddings)
        t_scale, t_shift = self.layer(t).chunk(2, dim=-1)
        torch.addcmul(t_shift[..., None, None], x, t_scale[..., None, None] + 1)  # x = (x * t_shift) + t_scale
        x = x * embeddings  # x = z((x * t_shift) + t_scale)
        return x


class Modulation2d(nn.Module):
    def __init__(self, feats_in, c_out):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(feats_in, c_out * 2),
            nn.SiLU()
        )

    def forward(self, input, embed):
        scales, shifts = self.layer(embed).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)
        self.weight.requires_grad_(False)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
