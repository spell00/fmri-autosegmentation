import random
import argparse
import torch.nn as nn
import torch
from ..utils.masked_layer import GatedConv3d
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from medicaltorch import transforms as mt_transforms
from torch.distributions import Beta
from torch.distributions.gamma import Gamma
import math
import numpy as np

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


def random_init(m, init_func=torch.nn.init.xavier_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


def log_gaussian(x, mu, log_var):
    """
    Returns the log pdf of a normal distribution parametrised
    by mu and log_var evaluated at x.

    :param x: point to evaluate
    :param mu: mean of distribution
    :param log_var: log variance of distribution
    :return: log N(x|µ,σ)
    """
    log_pdf = - 0.5 * torch.log(2 * torch.tensor(math.pi, requires_grad=True)) - log_var / 2 - (x - mu)**2 / (2 * torch.exp(log_var))
    return torch.sum(log_pdf, dim=-1)


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparameterize(self, mu):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        # std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(torch.ones_like(epsilon), epsilon)

        return z


class GaussianUnitSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianUnitSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features).to(device)
        self.log_var = nn.Linear(in_features, out_features).to(device)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparameterize(mu), mu, log_var


class GaussianBlock(nn.Module):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features, n_kernels=3):
        super(GaussianBlock, self).__init__()
        self.n_kernels = n_kernels
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_kernels)])
        self.log_var = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(n_kernels)])
        self.mu.apply(random_init)
        self.log_var.apply(random_init)

    def forward(self, x):
        mu = torch.stack([self.mu[i](x) for i in range(self.n_kernels)])
        log_var = torch.stack([F.softplus(self.log_var[i](x)) for i in range(self.n_kernels)])
        return x, mu, log_var

    def mle(self, x):
        return self.mu(x)

class BetaSample(nn.Module):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(BetaSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = nn.Linear(in_features, out_features).to(device)
        self.b = nn.Linear(in_features, out_features).to(device)

    def forward(self, x):
        a = self.a(x)
        b = F.softplus(self.b(x))
        pdf = torch._standard_gamma(a + b) * (x ** (a - 1)) * ((1 - x) ** (b - 1)) / (torch._standard_gamma(a) + torch._standard_gamma(b))

        return pdf


class ResBlock3D(nn.Module):
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


class ResBlockDeconv3D(nn.Module):
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


class ConvResnet3D(nn.Module):
    def __init__(self,
                 params,
                 n_classes,
                 is_bayesian=False,
                 ):
        super().__init__()
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.maxpool = nn.MaxPool3d(params['maxpool'], return_indices=False)

        self.is_bayesian = is_bayesian
        if is_bayesian:
            self.GaussianSample = GaussianUnitSample(37, 37)
        self.GaussianBlock = GaussianBlock(1, 1, n_kernels=len(params['kernel_sizes']))
        self.device = device
        self.conv_layers = nn.ModuleList()
        self.deconv_layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = nn.ModuleList()
        self.resconv = nn.ModuleList()
        self.activation = params['activation']()

        self.n_res = params['n_res']

        self.resblocks = params['resblocks']
        self.has_dense = params['has_dense']
        self.batchnorm = params['batchnorm']
        self.a_dim = None
        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(params['in_channels'], params['out_channels'],
                                                                        params['kernel_sizes'], params['strides'],
                                                                        params['dilatations'], params['padding'])):
            if not params['gated']:
                self.conv_layers += [
                    torch.nn.Conv3d(in_channels=ins,
                                    out_channels=outs,
                                    kernel_size=ksize,
                                    stride=stride,
                                    padding=pad,
                                    dilation=dilats,
                                    )
                ]
            else:
                self.conv_layers += [
                    GatedConv3d(input_channels=ins,
                                output_channels=outs,
                                kernel_size=ksize,
                                stride=stride,
                                padding=pad,
                                dilation=dilats,
                                activation=nn.Tanh()
                                )]
            if params['resblocks'] and i != 0:
                for _ in range(params['n_res']):
                    self.resconv += [ResBlock3D(ins, outs, params['activation'], device)]
            self.bns += [nn.BatchNorm3d(num_features=outs)]
        self.dropout3d = nn.Dropout3d(params['dropout'])
        self.dense1 = torch.nn.Linear(in_features=params['out_channels'][-1], out_features=32)
        self.dense1_bn = nn.BatchNorm1d(num_features=32)
        self.dense2 = torch.nn.Linear(in_features=32 + 5, out_features=n_classes)  # 5 parameters added here
        self.dense2_bn = nn.BatchNorm1d(num_features=n_classes)
        self.dropout = nn.Dropout(params['dropout'])
        self.log_softmax = torch.nn.functional.log_softmax

    def random_init(self, init_method=nn.init.xavier_uniform_):
        print("Random init")
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_method(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x, patient_info):
        j = 0
        for i in range(len(self.conv_layers)):
            if self.resblocks and i != 0:
                for _ in range(self.n_res):
                    x = self.resconv[j](x)
                    if self.batchnorm:
                        if x.shape[0] != 1:
                            x = self.bns[i - 1](x)
                    x = self.dropout3d(x)
                    j += 1
            x = self.conv_layers[i](x)
            if self.batchnorm:
                if x.shape[0] != 1:
                    x = self.bns[i](x)
            x = self.dropout3d(x)
            x = self.activation(x)
            x = self.maxpool(x)

        z = x.squeeze()
        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        z = self.dense1(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense1_bn(z)
        z = self.activation(z)
        z = self.dropout(z)
        z = torch.cat([z, patient_info], dim=1)
        if self.is_bayesian:
            z, _, _ = self.GaussianSample.float()(z)
        z = self.dense2(z)
        z = torch.sigmoid_(z)
        z, mu, log_var = self.GaussianBlock.float()(z)

        # if self.batchnorm:
        #     if z.shape[0] != 1:
        #         z = self.dense2_bn(z)
        # z = self.dropout(z)
        return z, mu, log_var

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
