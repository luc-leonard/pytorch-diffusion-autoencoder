from torch import nn

from model.modules.embeddings import Modulation2d, Identity


# cf appendix A
class ResConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, is_last=False, groups=32, timestep_embedding=None, embeddings_dim=None):
        super().__init__()
        self.skip = Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)

        self.gn_1 = nn.GroupNorm(groups, c_in)
        self.silu = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.conv_1 = nn.Conv2d(c_in, c_mid, 3, padding=1)
        self.gn_2 = nn.GroupNorm(groups, c_mid, affine=False)

        if timestep_embedding:
            self.modulation_1 = Modulation2d(timestep_embedding, c_mid)
        if embeddings_dim:
            self.embedding_mlp = ResLinearBlock(embeddings_dim, c_mid, c_mid, is_last=True)

        self.conv_2 = nn.Conv2d(c_mid, c_out, 3, padding=1)


    def forward(self, input, t=None, embedding=None):
        x = self.gn_1(input)
        x = self.silu(x)
        x = self.conv_1(x)
        x = self.gn_2(x)

        if t is not None:
            x = self.modulation_1(x, t)
        if embedding is not None:
            embedding = self.embedding_mlp(embedding)[:, :, None, None]
            x = x * embedding

        x = self.silu(x)
        x = self.dropout(x)
        x = self.conv_2(x)
        return x + self.skip(input)


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__([
            nn.Linear(f_in, f_mid),
            nn.SiLU(inplace=True),
            nn.Linear(f_mid, f_out),
            nn.SiLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)
