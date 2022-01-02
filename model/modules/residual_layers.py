from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return self.main(input) + self.skip(input)


class ResConvBlock(ResidualBlock):
    def __init__(self, c_in, c_mid, c_out, is_last=False):
        skip = None if c_in == c_out else nn.Conv2d(c_in, c_out, 1, bias=False)
        super().__init__(
            [
                nn.Conv2d(c_in, c_mid, 3, padding=1),
                nn.GroupNorm(1, c_mid, affine=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_mid, c_out, 3, padding=1),
                nn.GroupNorm(1, c_out, affine=False) if not is_last else nn.Identity(),
                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
            ],
            skip,
        )


class ResLinearBlock(ResidualBlock):
    def __init__(self, f_in, f_mid, f_out, is_last=False):
        skip = None if f_in == f_out else nn.Linear(f_in, f_out, bias=False)
        super().__init__(
            [
                nn.Linear(f_in, f_mid),
                nn.ReLU(inplace=True),
                nn.Linear(f_mid, f_out),
                nn.ReLU(inplace=True) if not is_last else nn.Identity(),
            ],
            skip,
        )
