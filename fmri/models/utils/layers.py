import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, activation=nn.ReLU, device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            activation(),
            nn.Conv3d(in_channel, channel, 3, padding=1),
            activation(),
            nn.Conv3d(channel, in_channel, 1),
        )
        self.conv.apply(random_init)

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class ResBlockDeconv(nn.Module):
    def __init__(self, in_channel, channel, activation=nn.ReLU, device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            activation(),
            nn.ConvTranspose3d(in_channel, channel, 1),
            activation(),
            nn.ConvTranspose3d(channel, in_channel, 3, padding=1),
        )
        self.conv.apply(random_init)

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


def random_init(m, init_func=torch.nn.init.xavier_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()
