from torch import nn

from model.modules.attention import SelfAttention2d
from model.modules.residual_layers import ResConvBlock
from model.openai.openai import AttentionBlock, Downsample, Upsample, ResBlock


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

        self.conv_in = ResBlock(
           channels=c_in,
            emb_channels=timestep_embeddings,
            dropout=0,
            out_channels=c_out,
            z_dim=embeddings_dim,

        )
        if attention:
            layers.append(AttentionBlock(c_out, num_heads=1, num_head_channels=-1))
        for i in range(inner_layers - 1):
            layers.append(
                ResBlock(
                    channels=c_out,
                    emb_channels=timestep_embeddings,
                    dropout=0,
                    out_channels=c_out,
                    z_dim=embeddings_dim,
                )
            )
            if attention:
                layers.append(SelfAttention2d(c_out, c_out // 64))

        if downsample:
            layers.append(Downsample(c_out, use_conv=True))
        if upsample:
            layers.append(Upsample(c_out, use_conv=True))
        self.main = nn.ModuleList(layers)

    def forward(self, x, t=None, embeddings=None):
        x = self.conv_in(x, t, embeddings)
        for layer in self.main:
            if isinstance(layer, ResConvBlock):
                x = layer(x, t, embeddings)
            else:
                x = layer(x)
        return x
