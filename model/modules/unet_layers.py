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
        embeddings_dim=None,
        timestep_embeddings=None,
    ):
        super().__init__()
        layers = []

        self.downsample = downsample

        ch = c_in
        for i in range(inner_layers):
            layers.append(
                ResBlock(
                    channels=ch,
                    emb_channels=timestep_embeddings,
                    dropout=0,
                    out_channels=c_out,
                    z_dim=embeddings_dim,
                )
            )
            ch = c_out
            if attention:
                layers.append(AttentionBlock(ch))

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
