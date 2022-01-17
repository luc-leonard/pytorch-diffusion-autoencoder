from torch import nn

from model.modules.embeddings import Modulation2d, Identity


class ResConvBlock(nn.Module):
    def __init__(self, c_in, c_mid, c_out, is_last=False, groups=32, embeddings_dim=None):
        super().__init__()
        self.skip = Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        self.conv_1 = nn.Conv2d(c_in, c_mid, 3, padding=1)
        self.gn_1 = nn.GroupNorm(groups, c_mid, affine=False)
        if embeddings_dim:
            self.modulation_1 = Modulation2d(embeddings_dim, c_mid)
        self.act_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(c_mid, c_out, 3, padding=1)
        if is_last:
            self.gn_2 = Identity()
            self.modulation_2 = Identity()
            self.act_out = Identity()
        else:
            self.gn_2 = nn.GroupNorm(groups, c_out, affine=False)
            if embeddings_dim:
                self.modulation_2 = Modulation2d(embeddings_dim, c_out)
            self.act_out = nn.ReLU(inplace=True)

    def forward(self, input, embedding=None):
        x = self.conv_1(input)
        x = self.gn_1(x)
        if embedding is not None:
            x = self.modulation_1(x, embedding)
        x = self.act_1(x)
        x = self.conv_2(x)
        x = self.gn_2(x)
        if embedding is not None:
            x = self.modulation_2(x, embedding)
        x = self.act_out(x)
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
            nn.ReLU(inplace=True),
            nn.Linear(f_mid, f_out),
            nn.ReLU(inplace=True) if not is_last else nn.Identity(),
        ], skip)
