import torch
from torch import nn

from model.modules.embeddings import FourierFeatures
from model.modules.unet_layers import UNetLayer


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_hidden_channels=128,
        n_layers=4,
        timestep_embed=16,
        chan_multiplier=[],
        inner_layers=[],
        attention_layers=[],
        z_dim=1,
        size=None,
    ):

        super().__init__()
        print("UNet")
        self.size = size
        self.in_channels = in_channels
        self.timestep_embed = FourierFeatures(1, timestep_embed)

        self.input_projection = UNetLayer(
            in_channels,
            base_hidden_channels,
            inner_layers=3,
            downsample=False,
            embeddings_dim=timestep_embed + z_dim,
        )

        down_layers = []
        up_layers = []
        current_size = size[0]
        for level in range(n_layers - 1):
            print(f'resolution : {current_size} for level {level}. Attentions: {attention_layers[level]}')
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level],
                base_hidden_channels * chan_multiplier[level + 1],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=True,
                embeddings_dim=timestep_embed + z_dim,
            )
            current_size //= 2
            down_layers.append(layer)
        print(f'resolution: {current_size} for level {n_layers}. Attentions: {attention_layers[-1]}')
        down_layers.append(UNetLayer(
            base_hidden_channels * chan_multiplier[-1],
            base_hidden_channels * chan_multiplier[-1],
            inner_layers=inner_layers[-1],
            attention=attention_layers[-1],
            downsample=True,
            embeddings_dim=timestep_embed + z_dim,
        ))

        up_layers.append(UNetLayer(
            base_hidden_channels * chan_multiplier[-1] * 2,
            base_hidden_channels * chan_multiplier[-1],
            inner_layers=inner_layers[-1],
            attention=attention_layers[-1],
            upsample=True,
            embeddings_dim=timestep_embed + z_dim,
        ))

        for level in reversed(range(n_layers - 1)):
            current_size *= 2
            print(f'resolution: {current_size} for level {level}. Attentions: {attention_layers[level]}')
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level + 1] * 2,
                base_hidden_channels * chan_multiplier[level],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                upsample=True,
                embeddings_dim=timestep_embed + z_dim,
            )
            up_layers.append(layer)


        self.down = nn.ModuleList(down_layers)
        self.up = nn.ModuleList(up_layers)

        self.output_projection = UNetLayer(
            base_hidden_channels * chan_multiplier[0],
            out_channels,
            inner_layers=3,
            upsample=False,
            is_last=True,
            embeddings_dim=timestep_embed + z_dim,
        )
        with torch.no_grad():
            for param in self.parameters():
                param *= 0.5 ** 0.5

    def forward(self, x, timestep, additional_embed=None):
        if len(timestep.shape) == 2:
            timestep = timestep.squeeze(1)

        timestep_embed = self.timestep_embed(timestep[:, None])
        embed = torch.cat([timestep_embed, additional_embed], dim=1)
        x = self.input_projection(x, embed)
        skips = []
        for down in self.down:
            x = down(x, embed)
            skips.append(x)

        for up, skip in zip(self.up, skips[::-1]):
            x = up(torch.cat([x, skip], dim=1), embed)

        x = self.output_projection(x, embed)
        return x
