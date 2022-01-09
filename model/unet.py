import torch
from torch import nn

from model.modules.embeddings import FourierFeatures
from model.modules.unet_layers import UNetLayer


def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class UNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, base_hidden_channels=128, n_layers=4,
            timestep_embed=16,
            chan_multiplier=[], inner_layers=[], attention_layers=[], z_dim=1, size=None
    ):

        super().__init__()
        self.size = size
        self.in_channels = in_channels
        self.timestep_embed = FourierFeatures(1, timestep_embed)

        self.input_projection = UNetLayer(
            in_channels + timestep_embed + z_dim, base_hidden_channels, inner_layers=3, downsample=False
        )

        down_layers = []
        up_layers = []

        for level in range(n_layers - 1):
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level] + timestep_embed + z_dim,
                base_hidden_channels * chan_multiplier[level + 1],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=level > 0,
            )
            down_layers.append(layer)

        for level in reversed(range(n_layers - 1)):
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level + 1] * 2 + timestep_embed + z_dim,
                base_hidden_channels * chan_multiplier[level],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                upsample=level > 0,
            )
            up_layers.append(layer)

        self.down = nn.ModuleList(down_layers)
        self.up = nn.ModuleList(up_layers)

        self.output_projection = UNetLayer(
            base_hidden_channels * chan_multiplier[0] + timestep_embed + z_dim,
            out_channels,
            inner_layers=3,
            upsample=False,
            is_last=True,
        )

        with torch.no_grad():
            for param in self.parameters():
                param *= 0.5 ** 0.5

    def embed_with(self, x, *embeddings):
        for e in embeddings:
            if e is not None:
                x = torch.cat([x, expand_to_planes(e, x.shape)], dim=1)
        return x

    def forward(self, x, timestep, additional_embed=None):

        timestep_embed = self.timestep_embed(timestep[:, None])
        x = self.input_projection(self.embed_with(x, timestep_embed, additional_embed))
        skips = []
        for down in self.down:
            x = down(self.embed_with(x, timestep_embed, additional_embed))
            skips.append(x)

        for up, skip in zip(self.up, skips[::-1]):
            x = up(torch.cat([self.embed_with(x, timestep_embed, additional_embed), skip], dim=1))

        x = self.output_projection(self.embed_with(x, timestep_embed, additional_embed))
        return x