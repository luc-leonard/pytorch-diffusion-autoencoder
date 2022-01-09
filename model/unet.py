from torch import nn
import torch

from model.latent_encoder import LatentEncoder
from model.modules.embeddings import FourierFeatures
from model.modules.unet_layers import UNetLayer


class UNet(nn.Module):
    def __init__(
        self, in_channels=3, out_channels=3, base_hidden_channels=128, n_layers=4,
            timestep_embed=16,
            chan_multiplier=[], inner_layers=[], attention_layers=[], z_dim=1
    ):
        super().__init__()
        self.timestep_embed = FourierFeatures(1, timestep_embed)

        self.input_projection = UNetLayer(
            in_channels + timestep_embed + z_dim, base_hidden_channels, inner_layers=3, downsample=False
        )

        down_layers = []
        up_layers = []

        for level in range(n_layers - 1):
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level] + timestep_embed + z_dim,
                base_hidden_channels * chan_multiplier[level + 1],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                downsample=level > 0,
            )
            down_layers.append(layer)

        for level in reversed(range(n_layers - 1)):
            layer = UNetLayer(
                base_hidden_channels * chan_multiplier[level + 1] * 2 + timestep_embed + z_dim,
                base_hidden_channels * chan_multiplier[level],
                inner_layers=inner_layers[level],
                attention=attention_layers[level],
                upsample=level > 0,
            )
            up_layers.append(layer)

        self.down = nn.ModuleList(down_layers)
        self.up = nn.ModuleList(up_layers)

        self.output_projection = UNetLayer(
            base_hidden_channels * chan_multiplier[0] + timestep_embed + z_dim,
            out_channels,
            inner_layers=3,
            upsample=False,
            is_last=True,
        )

        with torch.no_grad():
            for param in self.parameters():
                param *= 0.5 ** 0.5

    def embed_with(self, x, *embeddings):
        for e in embeddings:
            x = torch.cat([x, expand_to_planes(e, x.shape)], dim=1)
        return x

    def forward(self, x, timestep_embed, additional_embed=None):

        x = self.input_projection(self.embed_with(x, timestep_embed, additional_embed))
        skips = []
        for down in self.down:
            x = down(self.embed_with(x, timestep_embed, additional_embed))
            skips.append(x)

        for up, skip in zip(self.up, skips[::-1]):
            x = up(torch.cat([self.embed_with(x, timestep_embed, additional_embed), skip], dim=1))

        x = self.output_projection(self.embed_with(x, timestep_embed, additional_embed))
        return x



def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])


class DiffusionModel(nn.Module):
    def __init__(
        self,
        timestep_embed=16,
        size=None,
        in_channels=1,
        out_channels=1,
        base_hidden_channels=32,
        n_layers=2,
        chan_multiplier=[],
        inner_layers=[],
        attention_layers=[],
    ):
        super().__init__()
        self.size = size
        self.in_channel = in_channels

        self.timestep_embed = FourierFeatures(1, timestep_embed)

        self.unet = UNet(
            in_channels=in_channels + timestep_embed,
            out_channels=out_channels,
            base_hidden_channels=base_hidden_channels,
            n_layers=n_layers,
            chan_multiplier=chan_multiplier,
            inner_layers=inner_layers,
            attention_layers=attention_layers,
        )

    def forward(self, x, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        x = torch.cat([x, timestep_embed], dim=1)
        x = self.unet(x)
        return x


class ClassConditionedDiffusionModel(nn.Module):
    def __init__(
        self,
        timestep_embed=16,
        size=None,
        num_classes=10,
        in_channels=1,
        out_channels=1,
        base_hidden_channels=32,
        n_layers=2,
        chan_multiplier=[],
        inner_layers=[],
        attention_layers=[],
    ):
        super().__init__()
        self.size = size
        self.in_channel = in_channels

        self.timestep_embed = FourierFeatures(1, timestep_embed)
        if num_classes > 0:
            self.class_embed = nn.Embedding(num_classes, timestep_embed)
        else:
            self.class_embed = None

        self.unet = UNet(
            in_channels=in_channels + timestep_embed,
            out_channels=out_channels,
            base_hidden_channels=base_hidden_channels,
            n_layers=n_layers,
            chan_multiplier=chan_multiplier,
            inner_layers=inner_layers,
            attention_layers=attention_layers,
        )

    def forward(self, x, t, class_id=None):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)
        if self.class_embed is not None:
            class_embed = expand_to_planes(self.class_embed(class_id), x.shape)
            timestep_embed = timestep_embed + class_embed
        x = torch.cat([x, timestep_embed], dim=1)
        x = self.unet(x)
        return x


class AutoEncoderDiffusionModel(nn.Module):
    def __init__(
        self,
        timestep_embed=16,
        size=None,
        in_channels=1,
        out_channels=1,
        base_hidden_channels=32,
        n_layers=2,
        chan_multiplier=[],
        inner_layers=[],
        attention_layers=[],
        z_dim=3
    ):
        super().__init__()
        self.size = size
        self.in_channel = in_channels
        self.timestep_embed = FourierFeatures(1, timestep_embed)

        self.unet = UNet(
            in_channels=in_channels,
            out_channels=out_channels,
            base_hidden_channels=base_hidden_channels,
            n_layers=n_layers,
            chan_multiplier=chan_multiplier,
            inner_layers=inner_layers,
            attention_layers=attention_layers,
            z_dim=z_dim,
        )

    def forward(self, x, t, latent):
        timestep_embed = self.timestep_embed(t[:, None])
#        x = torch.cat([x, timestep_embed], dim=1)
        x = self.unet(x, timestep_embed, latent)
        return x


class PatchAutoEncoderDiffusionModel(nn.Module):
    def __init__(
        self,
        timestep_embed=16,
        size=None,
        in_channels=1,
        out_channels=1,
        base_hidden_channels=32,
        n_layers=2,
        chan_multiplier=[],
        inner_layers=[],
        attention_layers=[],
    ):
        super().__init__()
        self.size = size
        self.in_channel = in_channels
        self.timestep_embed = FourierFeatures(1, timestep_embed)

        self.unet = UNet(
            in_channels=in_channels + timestep_embed,
            out_channels=out_channels,
            base_hidden_channel=base_hidden_channels,
            n_layers=n_layers,
            chan_multiplier=chan_multiplier,
            inner_layers=inner_layers,
            attention_layers=attention_layers,
        )

    def forward(self, x, t, position_embed, latent):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), x.shape)

        x = torch.cat([x, timestep_embed], dim=1)
        x = self.unet(x)
        return x