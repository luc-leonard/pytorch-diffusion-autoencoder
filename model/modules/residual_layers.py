import torch
from torch import nn

from model.modules.embeddings import Identity


class AdaptiveGroupNormalization(nn.Module):
    def __init__(self, input_channels, timestep_channels, embedding_dim):
        super(AdaptiveGroupNormalization, self).__init__()
        self.z_mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.SiLU(),
            nn.Linear(embedding_dim, input_channels * 2),
            nn.SiLU(),
            nn.Linear(input_channels * 2, input_channels * 2),
            nn.SiLU(),
            nn.Linear(input_channels * 2, input_channels * 2),
            nn.SiLU(),
        )
        self.t_mlp = nn.Sequential(
            nn.Linear(timestep_channels, input_channels * 2), nn.SiLU()
        )

    def forward(self, x, t, embeddings):
        t_scale, t_shift = self.t_mlp(t).chunk(2, dim=-1)
        x = torch.addcmul(
            t_shift[..., None, None], x, t_scale[..., None, None] + 1
        )  # x = (x * t_shift) + t_scale

        z_scale, z_shift = self.z_mlp(embeddings).chunk(2, dim=-1)
        # x = z_scale * ((x * t_shift) + t_scale) + z_shift
        x = torch.addcmul(z_shift[..., None, None], x, z_scale[..., None, None] + 1)

        return x


# cf appendix A
class ResConvBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_mid,
        c_out,
        attention=False,
        groups=32,
        timestep_embedding=None,
        embeddings_dim=None,
    ):
        super().__init__()
        self.skip = (
            Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        )

        self.gn_1 = nn.GroupNorm(groups, c_in)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.conv_1 = nn.Conv2d(c_in, c_mid, 3, padding=1)
        self.gn_2 = nn.GroupNorm(groups, c_mid, affine=False)

        if timestep_embedding:
            self.ada_gn = AdaptiveGroupNormalization(
                c_mid, timestep_embedding, embeddings_dim
            )

        self.conv_2 = nn.Conv2d(c_mid, c_out, 3, padding=1)

    def forward(self, input, t=None, embedding=None):
        x = self.gn_1(input)
        x = self.silu(x)
        x = self.conv_1(x)
        x = self.gn_2(x)

        if t is not None:
            x = self.ada_gn(x, t, embedding)

        x = self.silu(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        x = self.silu(x)
        return x + self.skip(input)
