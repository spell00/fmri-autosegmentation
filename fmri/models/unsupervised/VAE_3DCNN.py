import torch
from ..utils.stochastic import GaussianSample
import torch.nn as nn
from ..utils.distributions import log_gaussian2, log_standard_gaussian
from ..utils.flow import NormalizingFlows, IAF, HouseholderFlow, ccLinIAF, SylvesterFlows
from ..utils.masked_layer import GatedConv3d, GatedConvTranspose3d, GatedConv2d, GatedConvTranspose2d
from ..utils.layers import ResBlock, ResBlockDeconv, random_init
from fmri.utils.quantizer import Quantize

in_channels = None
out_channels = None
kernel_sizes = None
strides = None


# TODO make it a 𝛽-VAE


class Autoencoder3DCNN(torch.nn.Module):
    def __init__(self, params):

        super(Autoencoder3DCNN, self).__init__()

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.params = params
        self.device = device

        self.indices = [torch.Tensor() for _ in range(len(in_channels))]
        self.indices2d = [torch.Tensor() for _ in range(3)]
        self.GaussianSample = GaussianSample(params['z_dim'], params['z_dim'])

        for i, (ins,
                outs,
                ksize,
                stride,
                dilats,
                pad,
                ) in enumerate(
            zip(params['in_channels'],
                params['out_channels'],
                params['kernel_sizes'],
                params['strides'],
                params['dilatations'],
                params['paddings'],
                )):
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
                    self.resconv += [ResBlock(ins, outs, params['activation'])]
            self.bns += [nn.BatchNorm3d(num_features=outs)]
            self.activations += [params['activation']]
        self.conv_layers2d = []
        if not params['gated']:
            for _ in range(3):
                self.conv_layers2d += [torch.nn.Conv2d(in_channels=params['in_channels'][-1],
                                                       out_channels=params['out_channels'][-1],
                                                       kernel_size=4,
                                                       stride=params['stride'][-1],
                                                       padding=params['padding'][-1],
                                                       dilation=params['dilatation'][-1],
                                                       )]

        else:
            for _ in range(3):
                self.conv_layers2d += [GatedConv2d(input_channels=ins,
                                                   output_channels=outs,
                                                   kernel_size=ksize,
                                                   stride=stride,
                                                   padding=pad,
                                                   dilation=dilats,
                                                   activation=nn.Tanh()
                                                   )]
        for i, (ins, outs, ksize, stride, dilats, pad) in enumerate(zip(reversed(params['out_channels'][-1]),
                                                                        reversed(params['in_channels'][-1]),
                                                                        params['kernel_sizes_deconv'][-1],
                                                                        params['strides_deconv'],
                                                                        params['dilatations_deconv'],
                                                                        params['padding_deconv']
                                                                        )):
            if not params['gated']:
                self.deconv_layers += [torch.nn.ConvTranspose3d(in_channels=ins, out_channels=outs,
                                                                kernel_size=ksize, padding=pad, stride=stride,
                                                                dilation=dilats)]

            else:
                self.deconv_layers += [GatedConvTranspose3d(input_channels=ins, output_channels=outs,
                                                            kernel_size=ksize,
                                                            stride=stride, padding=pad, dilation=dilats,
                                                            activation=nn.Tanh()
                                                            )]

            if params['resblocks'] and i != 0:
                for _ in range(params['']):
                    self.resdeconv += [ResBlockDeconv(ins, outs, params['activation'])]

            self.bns_deconv += [nn.BatchNorm3d(num_features=outs)]
            self.activations_deconv += [params['activation']()]
        self.deconv_layers2d = []
        if not params['gated']:
            for _ in range(3):
                self.deconv_layers2d += [torch.nn.ConvTranspose2d(in_channels=params['in_channels'],
                                                                  out_channels=params['out_channels'],
                                                                  kernel_size=4,
                                                                  stride=params['stride'],
                                                                  padding=0,
                                                                  dilation=params['dilatation'],
                                                                  )]
        else:
            self.deconv_layers2d = GatedConvTranspose2d(params['in_channels'][-1],
                                                        params['out_channels'][-1],
                                                        4,
                                                        params['stride'][-1],
                                                        0,
                                                        params['dilatation'][-1],
                                                        activation=nn.Tanh()
                                                        )
        self.dense1 = torch.nn.Linear(in_features=params['out_channels'][-1], out_features=params['z_dim'])
        self.dense2 = torch.nn.Linear(in_features=params['z_dim'], out_features=params['out_channels'][-1])
        self.dense_bn[0] = nn.BatchNorm1d(num_features=params['z_dim'])
        self.dense_bn[1] = nn.BatchNorm1d(num_features=params['out_channels'][-1])
        self.dropout3d = nn.Dropout3d(params['dropout'])
        self.dropout2d = nn.Dropout2d(params['dropout'])
        self.dropout = nn.Dropout(params['dropout'])
        self.max_pool = nn.MaxPool3d(params['max_pool'], return_indices=True)
        self.max_pool2d = nn.MaxPool2d(params['max_pool'], return_indices=True)
        self.max_unpool = nn.MaxUnpool3d(params['max_pool'])
        self.max_unpool2d = nn.MaxUnpool2d(params['max_pool'])
        self.conv_layers = nn.ModuleList(self.conv_layers)
        self.conv_layers2d = nn.ModuleList(self.conv_layers2d)
        self.deconv_layers = nn.ModuleList(self.deconv_layers)
        self.deconv_layers2d = nn.ModuleList(self.deconv_layers2d)
        self.bns = nn.ModuleList(self.bns)
        self.bns_deconv = nn.ModuleList(self.bns_deconv)
        self.resconv = nn.ModuleList(self.resconv)
        self.resdeconv = nn.ModuleList(self.resdeconv)

        self.flow_type = params['flow_type']
        self.n_flows = params['n_flows']
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[params['z_dim']], n_flows=params['z_dim'])
        if self.flow_type == "hf":
            self.flow = HouseholderFlow(in_features=[params['z_dims']], auxiliary=False, n_flows=params['n_flows'], h_last_dim=params['z_dim'])
        if self.flow_type == "iaf":
            self.flow = IAF(params['z_dim'], n_flows=params['n_flows'], num_hidden=params['n_flows'], h_size=params['z_dim'], forget_bias=1., conv3d=False)
        if self.flow_type == "ccliniaf":
            self.flow = ccLinIAF(in_features=[params['z_dim']], auxiliary=False, n_flows=params['n_flows'], h_last_dim=params['z_dim'])
        if self.flow_type == "o-sylvester":
            self.flow = SylvesterFlows(in_features=[params['z_dim']], flow_flavour='o-sylvester', n_flows=params['n_flows'], h_last_dim=None)
        if self.flow_type == "quantizer":
            self.flow = Quantize(params['z_dim'], self.n_embed)

    def random_init(self, init_func=torch.nn.init.xavier_uniform_):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                init_func(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()

    def _kld(self, z, q_param, h_last=None, p_param=None):
        if len(z.shape) == 1:
            z = z.view(1, -1)
        if (self.flow_type == "nf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z)
            qz = log_gaussian2(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type == "iaf") and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z, log_det_z = self.flow(z, h_last)
            qz = log_gaussian2(z, mu, log_var) - sum(log_det_z)
            z = f_z
        elif (self.flow_type in ['hf', 'ccliniaf']) and self.n_flows > 0:
            (mu, log_var) = q_param
            f_z = self.flow(z, h_last)
            qz = log_gaussian2(z, mu, log_var)
            z = f_z
        elif self.flow_type in ["o-sylvester", "h-sylvester", "t-sylvester"] and self.n_flows > 0:
            mu, log_var, r1, r2, q_ortho, b = q_param
            f_z = self.flow(z, r1, r2, q_ortho, b)
            qz = log_gaussian2(z, mu, log_var)
            z = f_z
        else:
            (mu, log_var) = q_param
            qz = log_gaussian2(z, mu, log_var)
        if p_param is None:
            pz = log_standard_gaussian(z)
        else:
            (mu, log_var) = p_param
            pz = log_gaussian2(z, mu, log_var)

        kl = qz - pz

        return kl

    def encoder(self, x):
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
            x = self.activations[i](x)
            x, self.indices[i] = self.max_pool(x)
        # not using .squeeze because it must not squeeze the first dimension when batch size is 1
        x = x.squeeze(-1)

        for j in range(len(self.conv_layers2d)):
            x = self.conv_layers2d[j](x)
            # if self.batchnorm:
            #     if x.shape[0] != 1:
            #         x = self.bns2d[i](x)
            x = self.dropout2d(x)
            x = self.activations[i](x)
            x, self.indices2d[j] = self.max_pool2d(x)

        z = x.squeeze(2).squeeze(2)
        if self.has_dense:
            z = self.dense1(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense1_bn(z)
            z = self.dropout(z)
            z = self.activation(z)
        return z

    def decoder(self, z):
        if self.has_dense:
            z = self.dense2(z)
            if self.batchnorm:
                if z.shape[0] != 1:
                    z = self.dense2_bn(z)
            z = self.dropout(z)
            z = self.activation_deconv(z)

        j = 0
        x = z.unsqueeze(2).unsqueeze(3)
        for i in range(len(self.deconv_layers2d)):
            ind = self.indices2d[len(self.indices2d) - 1 - i]
            x = self.max_unpool2d(x, ind)
            x = self.deconv_layers2d[i](x)
            # if self.batchnorm:
            #     if x.shape[0] != 1:
            #         x = self.bns_deconv[i](x)
            x = self.dropout2d(x)
            x = self.activations_deconv[i](x)
        x = x.unsqueeze(4)
        for i in range(len(self.deconv_layers)):
            if self.resblocks and i != 0:
                for _ in range(self.n_res):
                    x = self.resdeconv[j](x)
                    if self.batchnorm:
                        if x.shape[0] != 1:
                            x = self.bns_deconv[i - 1](x)
                    x = self.dropout3d(x)
                    j += 1
            ind = self.indices[len(self.indices) - 1 - i]
            x = self.max_unpool(x, ind)
            x = self.deconv_layers[i](x)
            if i < len(self.deconv_layers) - 1:
                if self.batchnorm:
                    if x.shape[0] != 1:
                        x = self.bns_deconv[i](x)
                x = self.dropout3d(x)
                x = self.activations_deconv[i](x)

        if len(x.shape) == 3:
            x.unsqueeze(0)
        x = torch.sigmoid(x)
        return x

    def forward(self, x, mle=False):
        kl = 0
        z = self.encoder(x)
        if not mle:
            z, mu, log_var = self.GaussianSample(z)

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
