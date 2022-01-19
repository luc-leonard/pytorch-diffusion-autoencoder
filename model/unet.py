import logging

import torch
from torch import nn

from model.modules.embeddings import FourierFeatures
from model.modules.residual_layers import ResLinearBlock
from model.modules.unet_layers import UNetLayer
from utils.torch import expand_to_planes


LOGGER = logging.getLogger(__name__)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        base_hidden_channels=128,
        n_layers=4,
        timestep_embed=16,
        mapping_timestep_embed=128,
        chan_multiplier=[],
        inner_layers=[],
        attention_layers=[],
        z_dim=1,
        cross_attention=False,
        size=None,
    ):

        super().__init__()
        self.size = size
        self.in_channels = in_channels

        # embedding layers
        self.timestep_embed = FourierFeatures(1, timestep_embed)


        down_layers = []
        up_layers = []
        self.input_projection = UNetLayer(
            in_channels,
            base_hidden_channels,
            inner_layers=3,
            downsample=False,
            embeddings_dim=z_dim,
            timestep_embeddings=timestep_embed,
            attention=False,
            groups=1
        )


        current_size = size[0]
        c_in = base_hidden_channels
        x_shapes = [c_in]

        for level in range(n_layers):
            LOGGER.info(f'resolution : {current_size} for level {level}. Attentions: {attention_layers[level]}')
            layer = UNetLayer(
                c_in=c_in,
                c_out=base_hidden_channels * chan_multiplier[level],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=True,
                embeddings_dim=z_dim,
                timestep_embeddings=timestep_embed,
            )
            c_in = base_hidden_channels * chan_multiplier[level]
            x_shapes.append(c_in)
            current_size //= 2
            down_layers.append(layer)

        self.middle_layer = UNetLayer(
            c_in=base_hidden_channels * chan_multiplier[-1],
            c_out=base_hidden_channels * chan_multiplier[-1],
            inner_layers=inner_layers[-1],
            embeddings_dim=z_dim,
            timestep_embeddings=timestep_embed,
            attention=True,
            downsample=False,
            upsample=False
        )
        x_shapes.pop()
        for level in reversed(range(n_layers)):
            current_size *= 2
            LOGGER.info(f'resolution: {current_size} for level {level}. Attentions: {attention_layers[level]}')
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level] * 2,
                x_shapes.pop(),
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                upsample=True,
                embeddings_dim=z_dim,
                timestep_embeddings=timestep_embed,
            )
            up_layers.append(layer)

        self.down = nn.ModuleList(down_layers)
        self.up = nn.ModuleList(up_layers)

        self.output_projection = UNetLayer(
            base_hidden_channels,
            out_channels,
            inner_layers=3,
            upsample=False,
            is_last=True,
            embeddings_dim=z_dim,
            groups=1,
            timestep_embeddings=timestep_embed,
        )

        with torch.no_grad():
            for param in self.parameters():
                param *= 0.5 ** 0.5

    def forward(self, x, timestep, additional_embed=None):
        if len(timestep.shape) == 2:
            timestep = timestep.squeeze(1)

        timestep_embed = self.timestep_embed(timestep[:, None])

        x = self.input_projection(x, timestep_embed, additional_embed)
        skips = []
        LOGGER.debug('before down', x.shape)
        for down in self.down:
            x = down(x, timestep_embed, additional_embed)
            skips.append(x)
            LOGGER.debug(x.shape)
        x = self.middle_layer(x, timestep_embed, additional_embed)
        LOGGER.debug('after mid', x.shape)
        for up, skip in zip(self.up, skips[::-1]):
            LOGGER.debug(x.shape, skip.shape)
            x = up(torch.cat([x, skip], dim=1), timestep_embed, additional_embed)

        x = self.output_projection(x, timestep_embed, additional_embed)
        return x
