from torch import nn

from model.modules.attention import SelfAttention2d
from model.modules.residual_layers import ResConvBlock


class UNetLayer(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        inner_layers: int,
        attention: bool = False,
        downsample: bool = False,
        upsample: bool = False,
        is_last=False,
    ):
        super().__init__()
        layers = []

        if downsample:
            assert upsample is False
            layers.append(nn.AvgPool2d(2))

        layers.append(ResConvBlock(c_in, c_out, c_out, is_last))
        if attention:
            layers.append(SelfAttention2d(c_out, c_out // 64))
        for i in range(inner_layers):
            layers.append(ResConvBlock(c_out, c_out, c_out, is_last))
            if attention:
                layers.append(SelfAttention2d(c_out, c_out // 64))

        if upsample:
            assert downsample is False
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
