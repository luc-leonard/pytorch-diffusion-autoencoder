from torch import nn

from model.modules.unet_layers import UNetLayer


class LatentEncoder(nn.Module):
    def __init__(
            self, in_channels=3, base_hidden_channel=128, n_layers=4,
            chan_multiplier=[], inner_layers=[], attention_layers=[], z_dim=3, **ignored
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
                downsample=True
                ,
            )
            down_layers.append(layer)
        self.net = nn.Sequential(*down_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(base_hidden_channel * chan_multiplier[-1], z_dim)

    def forward(self, x):
        x = self.input_projection(x)
        x = self.net(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.head(x)
        return x
