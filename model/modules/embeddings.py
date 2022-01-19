import torch
from torch import nn
import math


class Identity(nn.Module):
    def forward(self, x, *args):
        return x


class AdaptiveGroupNormalization(nn.Module):
    def __init__(self, input_channels, timestep_channels, embedding_dim):
        super(AdaptiveGroupNormalization, self).__init__()
        self.embeddings_mlp = nn.Linear(embedding_dim, input_channels)
        self.timestep_modulation = Modulation2d(timestep_channels, input_channels)

    def forward(self, x, t, embeddings):
        embeddings = self.embeddings_mlp(embeddings)
        x = self.timestep_modulation(x, t)
        x = x * embeddings
        return x


class Modulation2d(nn.Module):
    def __init__(self, feats_in, c_out):
        from model.modules.residual_layers import ResLinearBlock
        super().__init__()
        self.layer = ResLinearBlock(feats_in, c_out, c_out * 2, is_last=True)

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
