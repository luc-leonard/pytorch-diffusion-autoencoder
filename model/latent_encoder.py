import logging

from torch import nn

from model.modules.unet_layers import UNetLayer
import torch


LOGGER = logging.getLogger(__name__)

# use the openAI one for the moment
@DeprecationWarning
class LatentEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        base_hidden_channels,
        n_layers,
        chan_multiplier,
        inner_layers,
        attention_layers,
        z_dim,
        dropout=None,
        linear_layers=[],
    ):
        super(LatentEncoder, self).__init__()
        self.input_projection = UNetLayer(
            in_channels,
            base_hidden_channels,
            inner_layers=3,
            downsample=False,
            groups=1,
        )

        down_layers = []
        c_in = base_hidden_channels
        for level in range(n_layers):
            LOGGER.info(f"level {level}. Attentions: {attention_layers[level]}")
            layer = UNetLayer(
                c_in,
                base_hidden_channels * chan_multiplier[level],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=True,
                groups=32,
            )
            down_layers.append(layer)
            c_in = base_hidden_channels * chan_multiplier[level]
        self.layers = nn.ModuleList(down_layers)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        head_layers = []
        for i in range(1, len(linear_layers)):
            c_in = linear_layers[i - 1]
            c_out = linear_layers[i]
            head_layers.append(nn.Linear(c_in, c_out))
            head_layers.append(nn.SiLU(inplace=True))

        head_layers.append(nn.Linear(c_out, z_dim))
        head_layers.append(nn.SiLU(inplace=True))
        self.head = nn.Sequential(*head_layers)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

        with torch.no_grad():
            for param in self.parameters():
                param *= 0.5 ** 0.5

    def forward(self, x):
        x = self.input_projection(x)
        xs = []
        for layer in self.layers:
            x = layer(x)
            avg_x = x.mean(dim=(2, 3))
            xs.append(avg_x)

        x = torch.cat(xs, dim=1).squeeze(-1).squeeze(-1)
        x = self.head(x)
        return self.dropout(x)
