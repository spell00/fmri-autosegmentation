import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from fmri.models.utils.masked_layer import MaskedConv3d, MaskedLinear


class PlanarNormalizingFlow(nn.Module):
    """
    Planar normalizing flow [Rezende & Mohamed 2015].
    Provides a tighter bound on the ELBO by giving more expressive
    power to the approximate distribution, such as by introducing
    covariance between terms.
    """
    def __init__(self, in_features):
        super(PlanarNormalizingFlow, self).__init__()
        self.u = nn.Parameter(torch.randn(in_features))
        self.w = nn.Parameter(torch.randn(in_features))
        self.b = nn.Parameter(torch.ones(1))

    def forward(self, z):
        # Create uhat such that it is parallel to w
        uw = torch.dot(self.u, self.w)
        muw = -1 + F.softplus(uw)
        uhat = self.u + (muw - uw) * torch.transpose(self.w, 0, -1) / torch.sum(self.w ** 2)

        # Equation 21 - Transform z
        zwb = torch.mv(z, self.w) + self.b

        f_z = z + (uhat.view(1, -1) * torch.tanh(zwb).view(-1, 1))

        # Compute the Jacobian using the fact that
        # tanh(x) dx = 1 - tanh(x)**2
        psi = (1 - torch.tanh(zwb)**2).view(-1, 1) * self.w.view(1, -1)
        psi_u = torch.mv(psi, uhat)

        # Return the transformed output along
        # with log determninant of J
        logdet_jacobian = torch.log(torch.abs(1 + psi_u) + 1e-8)

        return f_z, logdet_jacobian


class HFlow(nn.Module):
    def __init__(self):
        super(HFlow, self).__init__()

    def forward(self, v, z):
        '''
        :param v: batch_size (B) x latent_size (L)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = z - 2* v v_T / norm(v,2) * z
        '''
        # v * v_T
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1) )  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L
        # v * v_T * z
        vvTz = torch.bmm(vvT, z.unsqueeze(2) ).squeeze(2) # A * z : batchdot( B x L x L * B x L x 1 ).squeeze(2) = (B x L x 1).squeeze(2) = B x L
        # calculate norm ||v||^2
        norm_sq = torch.sum(v * v, 1) # calculate norm-2 for each row : B x 1
        norm_sq = norm_sq.expand(v.size(1), norm_sq.size(0) ) # expand sizes : B x L
        # calculate new z
        z_new = z - 2 * vvTz / norm_sq.transpose(1, 0) # z - 2 * v * v_T  * z / norm2(v)
        return z_new


class linIAF(nn.Module):
    def __init__(self, z_dim):
        super().__init__()
        self.z_dim = z_dim

    def forward(self, l, z):
        '''
        :param L: batch_size (B) x latent_size^2 (L^2)
        :param z: batch_size (B) x latent_size (L)
        :return: z_new = L*z
        '''
        # L->tril(L)
        l_matrix = l.view(-1, self.z_dim, self.z_dim)  # resize to get B x L x L
        lt_mask = torch.tril(torch.ones(self.z_dim, self.z_dim), -1)  # lower-triangular mask matrix (1s in lower triangular part)
        I = Variable(torch.eye(self.z_dim, self.z_dim).expand(l_matrix.size(0), self.z_dim, self.z_dim))
        if self.cuda:
            lt_mask = lt_mask.cuda()
            I = I.cuda()
        lt_mask = Variable(lt_mask)
        lt_mask = lt_mask.unsqueeze(0).expand(l_matrix.size(0), self.z_dim, self.z_dim)  # 1 x L x L -> B x L x L
        lt = torch.mul(l_matrix, lt_mask) + I  # here we get a batch of lower-triangular matrices with ones on diagonal

        # z_new = L * z
        z_new = torch.bmm(lt, z.unsqueeze(2)).squeeze(2)  # B x L x L * B x L x 1 -> B x L

        return z_new


class CombinationL(nn.Module):
    def __init__(self, z_dim, n_combination):
        super().__init__()
        self.z_dim = z_dim
        self.n_combination = n_combination

    def forward(self, l, y):
        '''
        :param l: batch_size (B) x latent_size^2 * n_combination (L^2 * C)
        :param y: batch_size (B) x n_combination (C)
        :return: l_combination = y * L
        '''
        # calculate combination of Ls
        l_tensor = l.view(-1, self.z_dim ** 2, self.n_combination)  # resize to get B x L^2 x C
        y = y.unsqueeze(1).expand(y.size(0), self.z_dim ** 2, y.size(1))  # expand to get B x L^2 x C
        l_combination = torch.sum(l_tensor * y, 2).squeeze()
        return l_combination


class NormalizingFlows(nn.Module):
    """
    Presents a sequence of normalizing flows as a torch.nn.Module.
    """
    def __init__(self, in_features, n_flows=1, h_last_dim=None, flow_type=PlanarNormalizingFlow):
        self.h_last_dim = h_last_dim
        self.flows = []
        self.flows_a = []
        self.n_flows = n_flows
        self.flow_type = "nf"
        for i, features in enumerate(reversed(in_features)):
            self.flows += [nn.ModuleList([flow_type(features).cuda() for _ in range(n_flows)])]

        super(NormalizingFlows, self).__init__()

    def forward(self, z, i=0):
        log_det_jacobian = []
        flows = self.flows
        for flow in flows[i]:
            z, j = flow(z)
            log_det_jacobian.append(j)
        return z, sum(log_det_jacobian)


class HouseholderFlow(nn.Module):
    """
    Presents a sequence of normalizing flows as a torch.nn.Module.
    """
    def __init__(self, in_features, auxiliary, n_flows=1, h_last_dim=None, flow_type=HFlow, flow_flavour="hf"):
        super(HouseholderFlow, self).__init__()
        self.flow_flavour = flow_flavour
        self.v_layers = [[] for _ in range(len(in_features))]
        self.n_flows = n_flows
        self.flow_type = "hf"
        flows = []
        for i, features in enumerate(reversed(in_features)):
            flows += [flow_type().cuda()]
            v_layers = [nn.Linear(h_last_dim, features)] + [nn.Linear(features, features) for _ in range(n_flows)]
            self.v_layers[i] = nn.ModuleList(v_layers)
        if not auxiliary:
            self.flows = nn.ModuleList(flows)
        else:
            self.flows_a = nn.ModuleList(flows)

    def forward(self, z, h_last, auxiliary=False):
        self.cuda()
        v = {}
        z = {'0': z, '1': None}
        # Householder Flow:
        if self.n_flows > 0:
            v['1'] = self.v_layers[0][0].cuda()(h_last)
            if not auxiliary:
                z['1'] = self.flows[0](v['1'], z['0'])
            else:
                z['1'] = self.flows_a[0](v['1'], z['0'])

            for j in range(1, self.n_flows):
                v[str(j + 1)] = self.v_layers[0][j].cuda()(v[str(j)])
                if not auxiliary:
                    z[str(j + 1)] = self.flows[0](v[str(j + 1)], z[str(j)])
                else:
                    z[str(j + 1)] = self.flows_a[0](v[str(j + 1)], z[str(j)])

        return z[str(j + 1)]


class ccLinIAF(nn.Module):
    def __init__(self, in_features, n_flows=1, h_last_dim=None, flow_flavour="ccLinIAF", auxiliary=False, flow_type=linIAF):
        super().__init__()
        self.n_combination = n_flows
        self.n_flows = n_flows
        self.flow_flavour = flow_flavour
        flows = []
        combination_l = []
        encoder_y = []
        encoder_L = []

        for i, features in enumerate(list(reversed(in_features))):
            flows += [flow_type(features).cuda()]
            combination_l += [CombinationL(features, self.n_combination)]
            encoder_y += [nn.Linear(h_last_dim, self.n_combination)]
            encoder_L += [nn.Linear(h_last_dim, (features ** 2) * self.n_combination)]
        if not auxiliary:
            self.flows = nn.ModuleList(flows)
            self.combination_l = nn.ModuleList(combination_l)
            self.encoder_y = nn.ModuleList(encoder_y)
            self.encoder_L = nn.ModuleList(encoder_L)
        else:
            self.flows_a = nn.ModuleList(flows)
            self.combination_l_a = nn.ModuleList(combination_l)
            self.encoder_y_a = nn.ModuleList(encoder_y)
            self.encoder_L_a = nn.ModuleList(encoder_L)

        self.cuda()

    def forward(self, z, h_last, auxiliary=False, k=0):
        z = {'0': z, '1': None}
        if not auxiliary:
            l = self.encoder_L[k](h_last)
            y = F.softmax(self.encoder_y[k](h_last), dim=0)
            l_combination = self.combination_l[k](l, y)
            z['1'] = self.flows[k](l_combination, z['0'])
        else:
            l = self.encoder_L_a[k](h_last)
            y = F.softmax(self.encoder_y_a[k](h_last), dim=0)
            l_combination = self.combination_l_a[k](l, y)
            z['1'] = self.flows_a[k](l_combination, z['0'])

        return z['1']


class Sylvester(nn.Module):
    """
    Sylvester normalizing flow.
    """

    def __init__(self, num_ortho_vecs):

        super(Sylvester, self).__init__()

        self.num_ortho_vecs = num_ortho_vecs
        self.flow_type = "Sylvester"
        self.tanh = nn.Tanh()

        triu_mask = torch.triu(torch.ones(num_ortho_vecs, num_ortho_vecs), diagonal=1).unsqueeze(0)
        diag_idx = torch.arange(0, num_ortho_vecs).long()

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.tanh(x) ** 2

    def _forward(self, z, r1, r2, q_ortho, b, sum_ldj=True):
        """
        All flow parameters are amortized. Conditions on diagonals of R1 and R2 for invertibility need to be satisfied
        outside of this function. Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        :param z: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs)
        :param q_ortho: shape (batch_size, z_size , num_ortho_vecs)
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """
        # Amortized flow parameters
        z = z.unsqueeze(1)
        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        r1_hat = r1
        r2_hat = r2
        qr2 = torch.bmm(q_ortho, r2_hat.transpose(2, 1))
        qr1 = torch.bmm(q_ortho, r1_hat)
        r2qzb = torch.bmm(z, qr2) + b
        z = torch.bmm(self.tanh(r2qzb), qr1.transpose(2, 1)) + z
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, q_ortho, b, sum_ldj=True):

        return self._forward(z=zk, r1=r1, r2=r2, q_ortho=q_ortho, b=b, sum_ldj=sum_ldj)


class TriangularSylvester(nn.Module):
    """
    Sylvester normalizing flow with Q=P or Q=I.
    """

    def __init__(self, z_size):

        super(TriangularSylvester, self).__init__()

        self.z_size = z_size
        self.tanh = nn.Tanh()
        self.flow_type = "TriangularSylvester"

        diag_idx = torch.arange(0, z_size).long()
        self.register_buffer('diag_idx', diag_idx)

    def der_h(self, x):
        return self.der_tanh(x)

    def der_tanh(self, x):
        return 1 - self.tanh(x) ** 2

    def _forward(self, zk, r1, r2, b, auxiliary, permute_z=None, sum_ldj=True):
        """
        All flow parameters are amortized. conditions on diagonals of R1 and R2 need to be satisfied
        outside of this function.
        Computes the following transformation:
        z' = z + QR1 h( R2Q^T z + b)
        or actually
        z'^T = z^T + h(z^T Q R2^T + b^T)R1^T Q^T
        with Q = P a permutation matrix (equal to identity matrix if permute_z=None)
        :param zk: shape: (batch_size, z_size)
        :param r1: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param r2: shape: (batch_size, num_ortho_vecs, num_ortho_vecs).
        :param b: shape: (batch_size, 1, self.z_size)
        :return: z, log_det_j
        """

        # Amortized flow parameters
        zk = zk.unsqueeze(1)

        # Save diagonals for log_det_j
        diag_r1 = r1[:, self.diag_idx, self.diag_idx]
        diag_r2 = r2[:, self.diag_idx, self.diag_idx]

        if permute_z is not None:
            # permute order of z
            z_per = zk[:, :, permute_z]
        else:
            z_per = zk
        r2qzb = torch.bmm(z_per, r2.transpose(2, 1)) + b
        z = torch.bmm(self.tanh(r2qzb), r1.transpose(2, 1))

        if permute_z is not None:
            # permute order of z again back again
            z = z[:, :, permute_z]

        z += zk
        z = z.squeeze(1)

        # Compute log|det J|
        # Output log_det_j in shape (batch_size) instead of (batch_size,1)
        diag_j = diag_r1 * diag_r2
        diag_j = self.der_h(r2qzb).squeeze(1) * diag_j
        diag_j += 1.
        log_diag_j = diag_j.abs().log()

        if sum_ldj:
            log_det_j = log_diag_j.sum(-1)
        else:
            log_det_j = log_diag_j

        return z, log_det_j

    def forward(self, zk, r1, r2, b, auxiliary, permute_z, sum_ldj=True):

        return self._forward(zk, r1, r2, b, auxiliary, permute_z, sum_ldj)


class IAF(nn.Module):
    """
    PyTorch implementation of inverse autoregressive flows as presented in
    "Improving Variational Inference with Inverse Autoregressive Flow" by Diederik P. Kingma, Tim Salimans,
    Rafal Jozefowicz, Xi Chen, Ilya Sutskever, Max Welling.
    Inverse Autoregressive Flow with either MADE MLPs or Pixel CNNs. Contains several flows. Each transformation
     takes as an input the previous stochastic z, and a context h. The structure of each flow is then as follows:
     z <- autoregressive_layer(z) + h, allow for diagonal connections
     z <- autoregressive_layer(z), allow for diagonal connections
     :
     z <- autoregressive_layer(z), do not allow for diagonal connections.

     Note that the size of h needs to be the same as h_size, which is the width of the MADE layers.
     """

    def __init__(self, z_size, n_flows=2, num_hidden=0, h_size=50, forget_bias=1., conv3d=False):
        super(IAF, self).__init__()
        self.z_size = z_size
        self.n_flows = n_flows
        self.num_hidden = num_hidden
        self.tanh_size = h_size
        self.conv3d = conv3d
        if not conv3d:
            ar_layer = MaskedLinear
        else:
            ar_layer = MaskedConv3d
        self.activation = torch.nn.ELU
        # self.activation = torch.nn.ReLU

        self.forget_bias = forget_bias
        self.flows = []
        self.param_list = []

        # For reordering z after each flow
        flip_idx = torch.arange(self.z_size - 1, -1, -1).long()
        self.register_buffer('flip_idx', flip_idx)

        for k in range(n_flows):
            arch_z = [ar_layer(z_size, h_size), self.activation()]
            self.param_list += list(arch_z[0].parameters())
            z_feats = torch.nn.Sequential(*arch_z)
            arch_zh = []
            for j in range(num_hidden):
                arch_zh += [ar_layer(h_size, h_size), self.activation()]
                self.param_list += list(arch_zh[-2].parameters())
            zh_feats = torch.nn.Sequential(*arch_zh)
            linear_mean = ar_layer(h_size, z_size, diagonal_zeros=True)
            linear_std = ar_layer(h_size, z_size, diagonal_zeros=True)
            self.param_list += list(linear_mean.parameters())
            self.param_list += list(linear_std.parameters())

            if torch.cuda.is_available():
                z_feats = z_feats.cuda()
                zh_feats = zh_feats.cuda()
                linear_mean = linear_mean.cuda()
                linear_std = linear_std.cuda()
            self.flows.append((z_feats, zh_feats, linear_mean, linear_std))

        self.param_list = torch.nn.ParameterList(self.param_list)

    def forward(self, z, h_context):

        logdets = 0.
        for i, flow in enumerate(self.flows):
            if (i + 1) % 2 == 0 and not self.conv3d:
                # reverse ordering to help mixing
                z = z[:, self.flip_idx]

            h = flow[0](z)
            h = h + h_context
            h = flow[1](h)
            mean = flow[2](h)
            gate = torch.sigmoid(flow[3](h) + self.forget_bias)
            z = gate * z + (1 - gate) * mean
            logdets += torch.sum(gate.log().view(gate.size(0), -1), 1)
        return z, logdets


class SylvesterFlows(nn.Module):
    def __init__(self, in_features, flow_flavour, n_flows=1, h_last_dim=None, flow_type=Sylvester, auxiliary=None):
        super(SylvesterFlows, self).__init__()
        self.flows = []
        self.h_last_dim = h_last_dim
        self.z_dims = in_features
        self.z_dim = in_features[-1]
        self.n_flows = n_flows
        self.flow_type = "Sylvester"
        self.flow_flavour = flow_flavour

        # Normalizing flow layers
        for k in range(self.n_flows):
            for i in range(len(in_features)):
                flow_k = flow_type(self.n_flows)
                self.add_module('flow_' + str(k) + "_" + str(i) + "_" + str(auxiliary), flow_k)

    def forward(self, z, r1, r2, q_ortho, b, k=0, auxiliary=False):
        for i in range(self.n_flows):
            flow_name = 'flow_' + str(k) + "_" + str(i) + "_" + str(auxiliary)
            flow_k = getattr(self, flow_name)

            if self.flow_flavour == "o-sylvester":
                z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_ortho[k, :, :, :], b[:, :, :, k],
                                               auxiliary)
            if self.flow_flavour == "h-sylvester":
                z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], q_ortho, b[:, :, :, k], auxiliary)
            if self.flow_flavour == "t-sylvester":
                if k % 2 == 1:
                    # Alternate with reorderering z for triangular flow
                    permute_z = self.flip_idx
                else:
                    permute_z = None
                z_k, log_det_jacobian = flow_k(z[k], r1[:, :, :, k], r2[:, :, :, k], b[:, :, :, k], permute_z,
                                               auxiliary, sum_ldj=True)
            else:
                exit("Wrong flow_type")
            z.append(z_k)
            self.log_det_j += log_det_jacobian
        return z[-1], self.log_det_j