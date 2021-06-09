import torch
from ..utils.stochastic import GaussianSample
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ..utils.distributions import log_gaussian, log_standard_gaussian
from ..utils.flow import NormalizingFlows, IAF, HouseholderFlow, ccLinIAF, SylvesterFlows, TriangularSylvester
from ..utils.masked_layer import GatedConv1d, GatedConvTranspose1d

in_channels = None
out_channels = None
kernel_sizes = None
strides = None


def random_init(m, init_func=torch.nn.init.kaiming_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()


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


class Stochastic(nn.Module):
    """
    Base stochastic layer that uses the
    reparametrization trick [Kingma 2013]
    to draw a sample from a distribution
    parametrised by mu and log_var.
    """

    def reparameterize(self, mu, log_var):
        epsilon = Variable(torch.randn(mu.size()), requires_grad=False)

        if mu.is_cuda:
            epsilon = epsilon.cuda()

        # log_std = 0.5 * log_var
        # std = exp(log_std)
        std = log_var.mul(0.5).exp_()

        # z = std * epsilon + mu
        z = mu.addcmul(std, epsilon)

        return z


class GaussianSample(Stochastic):
    """
    Layer that represents a sample from a
    Gaussian distribution.
    """

    def __init__(self, in_features, out_features):
        super(GaussianSample, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_var = nn.Linear(in_features, out_features)

    def forward(self, x):
        mu = self.mu(x)
        log_var = F.softplus(self.log_var(x))

        return self.reparameterize(mu, log_var), mu, log_var


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel, activation=nn.ReLU, device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            activation(),
            nn.Conv1d(in_channel, channel, 3, padding=1),
            activation(),
            nn.Conv1d(channel, in_channel, 1),
        )
        self.conv.apply(random_init)
    def forward(self, input):
        out = self.conv.to(self.device)(input)
        out += input

        return out


class ResBlockDeconv(nn.Module):
    def __init__(self, in_channel, channel, activation=nn.ReLU, device='cuda'):
        super().__init__()
        self.device = device
        self.conv = nn.Sequential(
            activation(),
            nn.ConvTranspose1d(in_channel, channel, 1),
            activation(),
            nn.ConvTranspose1d(channel, in_channel, 3, padding=1),
        )
        self.conv.apply(random_init)

    def forward(self, input):
        out = self.conv.to(self.device)(input)
        out += input

        return out


class Autoencoder1DCNN(torch.nn.Module):
    def __init__(self,
                 z_dim,
                 maxpool,
                 in_channels,
                 out_channels,
                 kernel_sizes,
                 kernel_sizes_deconv,
                 strides,
                 strides_deconv,
                 dilatations,
                 dilatations_deconv,
                 padding,
                 padding_deconv,
                 batchnorm,
                 activation=torch.nn.GELU,
                 flow_type="nf",
                 n_flows=2,
                 n_res=3,
                 gated=True,
                 has_dense=True,
                 resblocks=False,
                 ):
        super(Autoencoder1DCNN, self).__init__()

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.device = device
        self.conv_layers = []
        self.deconv_layers = []
        self.bns = []
        self.resconv = []
        self.resdeconv = []
        self.bns_deconv = []
        self.indices = [torch.Tensor() for _ in range(len(in_channels))]
        self.GaussianSample = GaussianSample(z_dim, z_dim)
        self.activation = activation()

        self.n_res = n_res

        self.resblocks = resblocks
        self.has_dense = has_dense
        self.batchnorm = batchnorm
        self.a_dim = None
        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(in_channels, out_channels,
                                                                        kernel_sizes, strides,
                                                                        dilatations, padding)):
            if not gated:
                self.conv_layers += [
                    torch.nn.Conv1d(in_channels=ins,
                                    out_channels=outs,
                                    kernel_size=ksize,
                                    stride=stride,
                                    padding=pad,
                                    dilation=dilats,
                                    )
                ]
            else:
                self.conv_layers += [
                    GatedConv1d(input_channels=ins,
                                output_channels=outs,
                                kernel_size=ksize,
                                stride=stride,
                                padding=pad,
                                dilation=dilats,
                                activation=nn.Tanh()
                                )]
            if resblocks and i != 0:
                for _ in range(n_res):
                    self.resconv += [ResBlock(ins, outs, activation,device)]
            self.bns += [nn.BatchNorm1d(num_features=outs)]

        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(reversed(out_channels),
                                                                        reversed(in_channels),
                                                                        kernel_sizes_deconv,
                                                                        strides_deconv,
                                                                        dilatations_deconv,
                                                                        padding_deconv)):
            if not gated:
                self.deconv_layers += [torch.nn.ConvTranspose1d(in_channels=ins, out_channels=outs,
                                                                kernel_size=ksize, padding=pad, stride=stride,
                                                                dilation=dilats)]
            else:
                self.deconv_layers += [GatedConvTranspose1d(input_channels=ins, output_channels=outs,
                                                            kernel_size=ksize,
                                                            stride=stride, padding=pad, dilation=dilats,
                                                            activation=nn.Tanh()
                                                            )]
            if resblocks and i != 0:
                for _ in range(n_res):
                    self.resdeconv += [ResBlockDeconv(ins, outs, activation, device)]

            self.bns_deconv += [nn.BatchNorm1d(num_features=outs)]

        self.dense1 = torch.nn.Linear(in_features=out_channels[-1], out_features=z_dim)
        self.dense2 = torch.nn.Linear(in_features=z_dim, out_features=out_channels[-1])
        self.dense1_bn = nn.BatchNorm1d(num_features=z_dim)
        self.dense2_bn = nn.BatchNorm1d(num_features=out_channels[-1])
        self.dropout = nn.Dropout(0.5)
        self.maxpool = nn.MaxPool1d(maxpool, return_indices=True)
        self.maxunpool = nn.MaxUnpool1d(maxpool)
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.deconv_layers = nn.ModuleList(self.deconv_layers)
        self.bns = nn.ModuleList(self.bns)
        self.bns_deconv = nn.ModuleList(self.bns_deconv)
        self.resconv = nn.ModuleList(self.resconv)
        self.resdeconv = nn.ModuleList(self.resdeconv)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[z_dim], n_flows=n_flows)
        if self.flow_type == "hf":
            self.flow = HouseholderFlow(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim)
        if self.flow_type == "iaf":
            self.flow = IAF(z_dim, n_flows=n_flows, num_hidden=n_flows, h_size=z_dim, forget_bias=1., conv1d=False)
        if self.flow_type == "ccliniaf":
            self.flow = ccLinIAF(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim)
        if self.flow_type == "o-sylvester":
            self.flow = SylvesterFlows(in_features=[z_dim], flow_flavour='o-sylvester', n_flows=1, h_last_dim=None)

    def random_init(self, init_func=torch.nn.init.kaiming_uniform_):

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, h_last=None, p_param=None):
        if len(z.shape) == 1:
            z = z.view(1, -1)
        if (self.flow_type == "nf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type == "iaf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type in ['hf', 'ccliniaf']) and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z = self.flow(z, h_last)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        elif self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"] and self.n_flows > 0:
            mu, log_var, r1, r2, q_ortho, b = q_param
            f_z = self.flow(z, r1, r2, q_ortho, b)
            qz = log_gaussian(z, mu, log_var)
            z = f_z
        else:
            (mu, log_var) = q_param
            qz = log_gaussian(z, mu, log_var)
        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian(z, mu, log_var)

        kl = qz - pz

        return kl

    def encoder(self, x):
        j = 0
        for i in range(len(self.conv_layers)):
            if self.resblocks and i != 0:
                for _ in range(self.n_res):
                    if self.batchnorm:
                        if x.shape[0] != 1:
                            x = self.bns[i - 1].to(self.device)(x)
                    x = self.resconv[j](x)
                    j += 1
            x = self.conv_layers[i](x)
            if self.batchnorm:
                if x.shape[0] != 1:
                    x = self.bns[i].to(self.device)(x)
            x = self.activation(x)
            x = self.dropout(x)
            x, self.indices[i] = self.maxpool(x)

        z = x.squeeze()
        if self.has_dense:
            z = self.dense1(z)
            z = self.activation(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense1_bn(z)
            z = self.dropout(z)
        return z

    def decoder(self, z):
        if self.has_dense:
            z = self.dense2(z)
            z = self.activation(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense2_bn(z)
            z = self.dropout(z)

        j = 0
        x = z.unsqueeze(2)
        for i in range(len(self.deconv_layers)):
            if self.resblocks and i != 0:
                for _ in range(self.n_res):
                    if self.batchnorm:
                        if x.shape[0] != 1:
                            x = self.bns_deconv[i - 1].to(self.device)(x)
                    x = self.resdeconv[j](x)
                    j += 1
            ind = self.indices[len(self.indices) - 1 - i]
            x = self.maxunpool(x, ind)
            x = self.deconv_layers[i](x)
            if i < len(self.deconv_layers) - 1:
                if self.batchnorm:
                    if x.shape[0] != 1:
                        x = self.bns_deconv[i].to(self.device)(x)
                x = self.activation(x)
                x = self.dropout(x)

        if (len(x.shape) == 3):
            x.unsqueeze(0)
        # x = torch.sigmoid(x)
        return x

    def forward(self, x):

        x = self.encoder(x)
        z, mu, log_var = self.GaussianSample(x)

        # Kullback-Leibler Divergence
        kl = self._kld(z, (mu, log_var), x)

        if len(z.shape) == 1:
            z = z.unsqueeze(0)
        rec = self.decoder(z)
        return rec, kl

    def sample(self, z, y=None):
        """
        Given z ~ N(0, I) generates a sample from
        the learned distribution based on p_θ(x|z).
        :param z: (torch.autograd.Variable) Random normal variable
        :return: (torch.autograd.Variable) generated sample
        """
        return self.decoder(z)

    def get_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)

    def get_total_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.data.shape)
