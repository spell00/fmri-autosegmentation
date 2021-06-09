import torch
from ..utils.stochastic import GaussianSample
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from ..utils.distributions import log_gaussian2, log_standard_gaussian
from ..utils.flow import NormalizingFlows, IAF, HouseholderFlow, ccLinIAF, SylvesterFlows
from ..utils.masked_layer import GatedConv3d, GatedConvTranspose3d
from fmri.utils.quantizer import Quantize

in_channels = None
out_channels = None
kernel_sizes = None
strides = None


def random_init(m, init_func=torch.nn.init.xavier_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()




class Autoencoder3DMLP(torch.nn.Module):
    def __init__(self,
                 z_dim,
                 input_size,
                 batchnorm,
                 activation=torch.nn.ReLU,
                 flow_type="nf",
                 n_flows=2,
                 n_embed=2000,
                 dropout_val=0.5,
                 ):
        super(Autoencoder3DMLP, self).__init__()

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.n_embed = n_embed
        self.device = device
        self.bns = []
        self.activations = []
        self.activation = activation()
        self.GaussianSample = GaussianSample(z_dim, z_dim)

        self.batchnorm = batchnorm
        self.a_dim = None

        self.dense1 = torch.nn.Linear(in_features=input_size, out_features=512)
        self.dense11 = torch.nn.Linear(in_features=512, out_features=z_dim)
        self.dense2 = torch.nn.Linear(in_features=z_dim, out_features=512)
        self.dense22 = torch.nn.Linear(in_features=512, out_features=input_size)
        self.dense1_bn = nn.BatchNorm1d(num_features=512)
        self.dense11_bn = nn.BatchNorm1d(num_features=z_dim)
        self.dense2_bn = nn.BatchNorm1d(num_features=512)
        self.dense22_bn = nn.BatchNorm1d(num_features=input_size)
        self.dropout = nn.Dropout(dropout_val)
        self.bns = nn.ModuleList(self.bns)
        self.flow_type = flow_type
        self.n_flows = n_flows
        if self.flow_type == "nf":
            self.flow = NormalizingFlows(in_features=[z_dim], n_flows=n_flows)
        if self.flow_type == "hf":
            self.flow = HouseholderFlow(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim)
        if self.flow_type == "iaf":
            self.flow = IAF(z_dim, n_flows=n_flows, num_hidden=n_flows, h_size=z_dim, forget_bias=1., conv3d=False)
        if self.flow_type == "ccliniaf":
            self.flow = ccLinIAF(in_features=[z_dim], auxiliary=False, n_flows=n_flows, h_last_dim=z_dim)
        if self.flow_type == "o-sylvester":
            self.flow = SylvesterFlows(in_features=[z_dim], flow_flavour='o-sylvester', n_flows=1, h_last_dim=None)
        if self.flow_type == "quantizer":
            self.flow = Quantize(z_dim, self.n_embed)

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
        z = self.dense1(x.view(x.shape[0], -1))
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense1_bn(z)
        z = self.dropout(z)
        z = self.activation(z)
        z = self.dense11(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense11_bn(z)
        z = self.dropout(z)
        z = self.activation(z)
        return z

    def decoder(self, z):
        z = self.dense2(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense2_bn(z)
        z = self.dropout(z)
        z = self.activation(z)
        z = self.dense22(z)
        if self.batchnorm:
            if z.shape[0] != 1:
                z = self.dense22_bn(z)
        z = self.dropout(z)
        z = self.activation(z)

        x = torch.sigmoid(z)
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
