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
        embeddings_dim=None,
        groups=32,
        timestep_embeddings=None,
    ):
        super().__init__()
        layers = []

        self.downsample = downsample
        if downsample:
            self.avgpool = nn.AvgPool2d(2)

        self.conv_in = ResConvBlock(c_in, c_out, c_out, is_last, groups, timestep_embeddings, embeddings_dim)
        if attention:
            layers.append(SelfAttention2d(c_out, c_out // 64))
        for i in range(inner_layers):
            layers.append(ResConvBlock(c_out, c_out, c_out, is_last, groups, timestep_embeddings, embeddings_dim))
            if attention:
                layers.append(SelfAttention2d(c_out, c_out // 64))

        if upsample:
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
        self.main = nn.ModuleList(layers)

    def forward(self, x, t=None, embeddings=None):
        if self.downsample:
            x = self.avgpool(x)

        x = self.conv_in(x, t, embeddings)
        for layer in self.main:
            if isinstance(layer, ResConvBlock):
                x = layer(x, t, embeddings)
            else:
                x = layer(x)
        return x
