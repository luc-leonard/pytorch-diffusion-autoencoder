from math import sqrt

import torch.nn.functional as F
from torch.nn import Conv1d, Linear
from torch import nn
import torch

from model.openai.nn import timestep_embedding


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        """
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        """
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = Linear(64, residual_channels)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step):
        diffusion_step = self.diffusion_projection(diffusion_step)[..., None]
        y = x + diffusion_step
        y = self.dilated_conv(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        x = (x + residual) / sqrt(2.0)
        return x, skip


class DiffWave(nn.Module):
    def __init__(
        self, residual_layers, model_channels, dilation_cycle_length
    ):
        super().__init__()
        print("Building DiffWave model...")
        self.in_channels = 1
        self.out_channels = 1
        self.size = [22050]

        self.model_channels = model_channels

        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )
        self.input_projection = Conv1d(1, model_channels, 1)


        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(model_channels, 2 ** (i % dilation_cycle_length))
                for i in range(residual_layers)
            ]
        )

        self.skip_projection = Conv1d(
            64, 64, 1
        )
        self.output_projection = Conv1d(64, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, timesteps, latent):
        x = self.input_projection(audio)
        x = F.relu(x)

        diffusion_step = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        return x


class Encoder(nn.Module):
    def __init__(self, dummy):
        super().__init__()
        self.dummy = dummy

    def forward(self, x):
        return torch.randn(x.shape[0], 256)
