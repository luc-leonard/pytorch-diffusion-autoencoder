from torch import nn

from model.modules.unet_layers import UNetLayer


class LatentEncoder(nn.Module):
    def __init__(
            self, in_channels=3, base_hidden_channel=128, n_layers=4,
            chan_multiplier=[], inner_layers=[], attention_layers=[],
    ):
        super(LatentEncoder, self).__init__()
        self.input_projection = UNetLayer(
            in_channels, base_hidden_channel, inner_layers=3, downsample=False
        )
        down_layers = []
        for level in range(n_layers - 1):
            layer = UNetLayer(
                base_hidden_channel * chan_multiplier[level],
                base_hidden_channel * chan_multiplier[level + 1],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=level > 0,
            )
            down_layers.append(layer)
        self.net = nn.Sequential(*down_layers)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.net(x)
        return x
