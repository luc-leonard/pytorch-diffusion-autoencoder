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
        linear_layers=[]
    ):
        super(LatentEncoder, self).__init__()
        print('LatentEncoder')
        self.input_projection = UNetLayer(
            in_channels, base_hidden_channels, inner_layers=3, downsample=False
        )

        down_layers = []
        for level in range(n_layers - 1):
            print(f'level {level}. Attentions: {attention_layers[level]}')
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level],
                base_hidden_channels * chan_multiplier[level + 1],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=True,
            )
            down_layers.append(layer)
        print(f'level {n_layers - 1}. Attentions: {attention_layers[-1]}')
        down_layers.append(UNetLayer(
            base_hidden_channels * chan_multiplier[-1],
            base_hidden_channels * chan_multiplier[-1],
            inner_layers=inner_layers[-1],
            attention=attention_layers[-1],
            downsample=True,
        ))
        self.net = nn.Sequential(*down_layers)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        head_layers = []
        c_out = base_hidden_channels * chan_multiplier[-1]
        for i in range(len(linear_layers)):
            if i == 0:
                c_in = base_hidden_channels * chan_multiplier[-1]
            else:
                c_in = linear_layers[i - 1]
            head_layers.append(nn.Linear(c_in, linear_layers[i]))
            head_layers.append(nn.Mish())
            c_out = linear_layers[i]

        head_layers.append(nn.Linear(c_out, z_dim))
        self.head = nn.Sequential(*head_layers)
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
