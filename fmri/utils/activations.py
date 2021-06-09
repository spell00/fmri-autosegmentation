in_channels = None
out_channels = None
kernel_sizes = None
strides = None
from torch import nn
from torch.nn import functional as F

def swish(x):
    return x * x.sigmoid()


def mish(x):
    return x * F.softplus(x).tanh()


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return swish(x)


class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return mish(x)


