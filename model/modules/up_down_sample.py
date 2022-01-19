from torch import nn


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = self.up(x)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.AvgPool2d(2)
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = self.down(x)
        x = self.conv(x)
        return x