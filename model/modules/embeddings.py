import torch
from torch import nn
import math


class Identity(nn.Module):
    def forward(self, x, *args):
        return x


class Modulation2d(nn.Module):
    def __init__(self, feats_in, c_out):
        super().__init__()
        self.layer = nn.Linear(feats_in, c_out * 2, bias=False)

    def forward(self, input, embed):
        scales, shifts = self.layer(embed).chunk(2, dim=-1)
        return torch.addcmul(shifts[..., None, None], input, scales[..., None, None] + 1)


class SimpleModulation2d(nn.Module):
    def __init__(self, feats_in, c_out):
        super().__init__()
        self.layer = nn.Linear(feats_in, c_out, bias=False)

    def forward(self, input, embed):
        embed = self.layer(embed)
        return input + embed

class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)
        self.weight.requires_grad_(False)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)
