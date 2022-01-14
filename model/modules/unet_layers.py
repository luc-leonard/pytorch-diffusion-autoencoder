from torch import nn

from model.modules.attention import SelfAttention2d
from model.modules.residual_layers import ResConvBlock
from utils.torch import expand_to_planes


class CrossAttention(nn.Module):
    def __init__(self, in_channels, out_channels, query_size):
        super(CrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(in_channels, in_channels // 64, qd)
        self.linear_in = nn.Linear(in_channels, out_channels)

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
        cross_attention=False,
        groups=32,
    ):
        super().__init__()
        layers = []

        if embeddings_dim:
            if not cross_attention:
                self.embedding_mlp = nn.Sequential(
                    nn.Mish(),
                    # attention between embeddings and input ?
                    nn.Linear(embeddings_dim, c_out),
                )
            else:
                self.embedding_mlp = nn.ModuleList([
                    nn.Mish(),
                    nn.MultiheadAttention(embeddings_dim,  embeddings_dim // 64, kdim=c_in, vdim=c_in),
                    nn.Linear(embeddings_dim, c_out),
                ])
        self.downsample = downsample
        if downsample:
            self.avgpool = nn.AvgPool2d(2)

        self.conv_in = ResConvBlock(c_in, c_out, c_out, is_last, groups)
        if attention:
            layers.append(SelfAttention2d(c_out, c_out // 64))
        for i in range(inner_layers):
            layers.append(ResConvBlock(c_out, c_out, c_out, is_last, groups))
            if attention:
                layers.append(SelfAttention2d(c_out, c_out // 64))

        if upsample:
            layers.append(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
            )
        self.main = nn.Sequential(*layers)

    def forward(self, x, embeddings=None):
        if self.downsample:
            x = self.avgpool(x)

        x = self.conv_in(x)
        if embeddings is not None:
            h = self.embedding_mlp(embeddings)
            x = x + expand_to_planes(h, x.shape)
        x = self.main(x)
        return x
