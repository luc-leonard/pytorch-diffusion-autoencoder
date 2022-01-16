import torch
from torch import nn
import math


class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, query_size):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(in_channels, in_channels // 64, qd)
        self.linear_in = nn.Linear(in_channels, out_channels)


class Modulation2d(nn.Module):
    def __init__(self, feats_in, c_out):
        super().__init__()
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input, embed):
        scales, shifts = self.laye(embed).chunk(2, dim=-1)
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
