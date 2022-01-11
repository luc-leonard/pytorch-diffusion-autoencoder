from torch import nn

from model.modules.unet_layers import UNetLayer


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
    ):
        super(LatentEncoder, self).__init__()
        self.input_projection = UNetLayer(
            in_channels, base_hidden_channels, inner_layers=3, downsample=False
        )

        down_layers = []
        for level in range(n_layers - 1):
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level],
                base_hidden_channels * chan_multiplier[level + 1],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=True,
            )
            down_layers.append(layer)
        self.net = nn.Sequential(*down_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(base_hidden_channels * chan_multiplier[-1], z_dim)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()

    def forward(self, x):
        x = self.input_projection(x)
        x = self.net(x)
        x = self.avgpool(x)
        x = x.squeeze(-1).squeeze(-1)
        x = self.head(x)
        return self.dropout(x)
