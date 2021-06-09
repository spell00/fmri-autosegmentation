import torch
import torch.nn as nn
from torch.autograd import Variable
from fmri.models.utils.distributions import log_gaussian, log_standard_gaussian
from fmri.models.utils.flow import Sylvester, TriangularSylvester, SylvesterFlows
from fmri.models.unsupervised.VAE_1DCNN import Autoencoder1DCNN

def random_init(m, init_func=torch.nn.init.kaiming_uniform_):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
        init_func(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

class SylvesterVAE(Autoencoder1DCNN):
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
                 flow_type,
                 gated,
                 has_dense,
                 resblocks,
                 h_last,
                 n_flows,
                 n_res,
                 num_elements=3,
                 auxiliary=False,
                 a_dim=0,
                 ):
        super(SylvesterVAE, self).__init__(z_dim,
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
                                           flow_type=flow_type,
                                           n_flows=n_flows,
                                           n_res=n_res,
                                           gated=gated,
                                           has_dense=has_dense,
                                           resblocks=resblocks,
                                           )
        # Initialize log-det-jacobian to zero
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        self.device = device
        self.auxiliary = auxiliary
        self.log_det_j = 0.
        # Flow parameters
        self.flavor = SylvesterFlows
        self.h_last = h_last
        if type(z_dim) is not list:
            self.z_dim_last = z_dim
            self.z_dims = z_dims = [z_dim]
        else:
            self.z_dims = z_dims = z_dim
            self.z_dim_last = z_dims[-1]
        self.n_flows = n_flows
        self.num_elements = num_elements

        if type(self.a_dim) is not list:
            self.a_dims = [a_dim]  # for compatibility with ladder network
            self.a_dim = a_dim
        else:
            self.a_dims = a_dim
            self.a_dim = self.a_dims[-1]
        if type(self.z_dims) is list:
            assert (self.num_elements <= self.z_dim_last) and (self.num_elements > 0)
        else:
            self.z_dims = [z_dims]
            assert (self.num_elements <= self.z_dim_last) and (self.num_elements > 0)

        # Orthogonalization parameters
        if self.num_elements == self.z_dim_last and self.flow_flavour == "o-sylvester":
            self.cond = 1.e-5
        else:
            self.cond = 1.e-6

        self.steps = 100

        self.diag_activation = nn.Tanh()

        if self.flow_type == "o-sylvester":
            identity = torch.eye(self.num_elements, self.num_elements)
            triu_mask = torch.triu(torch.ones(self.num_elements, self.num_elements), diagonal=1)
            diag_idx = torch.arange(0, self.num_elements).long()
            self.register_buffer('diag_idx', diag_idx)

            self.amor_d = nn.Linear(self.h_last, self.n_flows * self.num_elements * self.num_elements)
            self.amor_diag1 = nn.Sequential(
                nn.Linear(self.h_last, self.n_flows * self.num_elements),
                self.diag_activation
            )
            self.amor_diag2 = nn.Sequential(
                nn.Linear(self.h_last, self.n_flows * self.num_elements),
                self.diag_activation
            )
            self.amor_diag1.apply(random_init)
            self.amor_diag2.apply(random_init)
            self.amor_b = nn.Linear(self.h_last, self.n_flows * self.num_elements)
            if len(self.z_dims) > 1:
                self.amor_q = [nn.Linear(self.h_last, self.n_flows * z_dim * self.num_elements) for z_dim in
                               self.z_dims]
            else:
                self.amor_q = nn.Linear(self.h_last, self.n_flows * self.z_dim_last * self.num_elements)

            if self.auxiliary:
                if len(self.a_dims) > 1:
                    self.amor_q_a = [nn.Linear(self.h_last, self.n_flows * a_dim * self.num_elements) for a_dim in
                                     self.a_dims]
                else:
                    try:
                        self.amor_q_a = nn.Linear(self.h_last, self.n_flows * self.a_dims[0] * self.num_elements)
                    except:
                        self.amor_q_a = nn.Linear(self.h_last, self.n_flows * self.a_dims[0][0] * self.num_elements)

                diag_idx_aux = torch.arange(0, self.z_dim_last).long()
                self.register_buffer('diag_idx_aux', diag_idx_aux)

        elif self.flow_type in ["h-sylvester", "t-sylvester"]:
            identity = torch.eye(self.z_dim_last, self.z_dim_last)
            # Add batch dimension
            identity = identity.unsqueeze(0)
            triu_mask = torch.triu(torch.ones(self.z_dim_last, self.z_dim_last), diagonal=1)
            diag_idx = torch.arange(0, self.z_dim_last).long()
            self.register_buffer('diag_idx', diag_idx)
            if self.auxiliary:
                diag_idx_aux = torch.arange(0, self.a_dim).long()
            self.amor_d = nn.Linear(self.h_last, self.n_flows * self.z_dim_last * self.z_dim_last)
            self.amor_diag1 = nn.Sequential(
                nn.Linear(self.h_last, self.n_flows * self.z_dim_last),
                self.diag_activation
            )
            self.amor_diag2 = nn.Sequential(
                nn.Linear(self.h_last, self.n_flows * self.z_dim_last),
                self.diag_activation
            )
            self.amor_diag1.apply(random_init)
            self.amor_diag2.apply(random_init)
            self.amor_b = nn.Linear(self.h_last, self.n_flows * self.z_dim_last)
            if self.flow_type == "h-sylvester":
                self.amor_q = [nn.Linear(self.h_last, self.n_flows * z_dim * self.num_elements) for z_dim in
                               self.z_dims]
            if self.auxiliary:
                identity_a = torch.eye(self.a_dim, self.a_dim)
                triu_mask_a = torch.triu(torch.ones(self.a_dim, self.a_dim), diagonal=1)
                self.amor_d_a = nn.Linear(self.h_last, self.n_flows * self.a_dim * self.a_dim)
                self.amor_diag1_a = nn.Sequential(
                    nn.Linear(self.h_last, self.n_flows * self.a_dim),
                    self.diag_activation
                )
                self.amor_diag2_a = nn.Sequential(
                    nn.Linear(self.h_last, self.n_flows * self.a_dim),
                    self.diag_activation
                )
                self.amor_b_a = nn.Linear(self.h_last, self.n_flows * self.a_dim)
                if self.flow_type == "h-sylvester":
                    self.amor_q_a = [nn.Linear(self.h_last, self.n_flows * a_dim * self.num_elements) for a_dim in
                                     self.a_dims]

        if self.flow_type == "t-sylvester":
            flip_idx = torch.arange(self.z_dim_last - 1, -1, -1).long()
            self.register_buffer('flip_idx', flip_idx)

        # Add batch dimension
        identity = identity.unsqueeze(0)
        # Put identity in buffer so that it will be moved to GPU if needed by any call of .cuda
        self.register_buffer('_eye', Variable(identity))

        if auxiliary and self.flow_type != "o-sylvester":
            identity_a = identity_a.unsqueeze(0)
            self.register_buffer('_eye_a', Variable(identity_a))
            self._eye_a.requires_grad = False
            triu_mask_a = triu_mask_a.unsqueeze(0).unsqueeze(3)
            self.register_buffer('triu_mask_a', Variable(triu_mask_a))

        self._eye.requires_grad = False

        # Masks needed for triangular R1 and R2.
        triu_mask = triu_mask.unsqueeze(0).unsqueeze(3)

        self.register_buffer('triu_mask', Variable(triu_mask))
        self.triu_mask.requires_grad = False
        if self.auxiliary and self.flow_type != "o-sylvester":
            self.register_buffer('diag_idx_aux', diag_idx_aux)
        # Normalizing flow layers
        for k in range(len(self.z_dims)):
            for i in range(self.n_flows):
                if self.flow_type == "o-sylvester":
                    flow_k = Sylvester(self.num_elements)
                    if self.auxiliary:
                        flow_k_a = Sylvester(self.num_elements)
                elif self.flow_type == "h-sylvester":
                    flow_k = Sylvester(self.z_dim_last)
                    if self.auxiliary:
                        flow_k_a = Sylvester(self.a_dim)
                elif self.flow_type == "t-sylvester":
                    flow_k = TriangularSylvester(self.z_dim_last)
                    if self.auxiliary:
                        flow_k_a = TriangularSylvester(self.a_dim)

                self.add_module('flow_' + str(k) + "_" + str(i) + "_" + str(False), flow_k)
                if self.auxiliary:
                    name_aux = 'flow_' + str(k) + "_" + str(i) + "_" + str(True)
                    self.add_module(name_aux, flow_k_a)

    def batch_construct_orthogonal(self, q, auxiliary, i=-1):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size * n_flows, z_dim_last * num_elements)
        :return: batches of orthogonalized matrices, shape: (batch_size * n_flows, z_dim_last, num_elements)
        """

        # Reshape to shape (n_flows * batch_size, z_dim_last * num_elements)
        if auxiliary:
            last = self.a_dims[-1]
        else:
            last = self.z_dims[-1]
        last_type = type(last)
        while last_type is list:
            last = last[0]
            last_type = type(last)

        q = q.view(-1, last * self.num_elements)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)
        amat = torch.div(q, norm)
        dim0 = amat.size(0)
        amat = amat.reshape(dim0, last, self.num_elements)

        max_norm = 0.

        # Iterative orthogonalization
        for s in range(self.steps):
            tmp = torch.bmm(amat.transpose(2, 1), amat)
            tmp = self._eye - tmp
            tmp = self._eye + 0.5 * tmp
            amat = torch.bmm(amat, tmp)

            # Testing for convergence
            test = torch.bmm(amat.transpose(2, 1), amat) - self._eye
            norms2 = torch.sum(torch.norm(test, p=2, dim=2) ** 2, dim=1)
            norms = torch.sqrt(norms2)
            max_norm = torch.max(norms).item()
            if max_norm <= self.cond:
                break

        if max_norm > self.cond:
            print('\nWARNING WARNING WARNING: orthogonalization not complete')
            print('\t Final max norm =', max_norm)

            print()

        # Reshaping: first dimension is batch_size
        amat = amat.view(-1, self.n_flows, last, self.num_elements)
        amat = amat.transpose(0, 1)

        return amat

    def batch_construct_householder_orthogonal(self, q, auxiliary):
        """
        Batch orthogonal matrix construction.
        :param q:  q contains batches of matrices, shape : (batch_size, n_flows * z_dim_last * num_elements)
        :return: batches of orthogonalized matrices, shape: (batch_size * n_flows, z_dim_last, z_dim_last)
        """

        # Reshape to shape (n_flows * batch_size * num_elements, z_dim_last)
        if auxiliary:
            last = self.a_dim
            _eye = self._eye_a
        else:
            last = self.z_dim_last
            _eye = self._eye

        q = q.view(-1, last)

        norm = torch.norm(q, p=2, dim=1, keepdim=True)  # ||v||_2
        v = torch.div(q, norm)  # v / ||v||_2

        # Calculate Householder Matrices
        vvT = torch.bmm(v.unsqueeze(2), v.unsqueeze(1))  # v * v_T : batch_dot( B x L x 1 * B x 1 x L ) = B x L x L

        amat = _eye - 2 * vvT  # NOTICE: v is already normalized! so there is no need to calculate vvT/vTv

        # Reshaping: first dimension is batch_size * n_flows
        amat = amat.view(-1, self.num_elements, last, last)

        tmp = amat[:, 0]
        for k in range(1, self.num_elements):
            tmp = torch.bmm(amat[:, k], tmp)

        amat = tmp.view(-1, self.n_flows, last, last)

        amat = amat.transpose(0, 1)
        return amat

    def encode(self, x, y, a, auxiliary, i=0):
        """
        Encoder that ouputs parameters for base distribution of z and flow parameters.
        """

        if auxiliary and self.flow_type != "o-sylvester":
            if self.flow_type != "t-sylvester":
                amor_q = self.amor_q_a
            last = self.a_dim
            amor_b = self.amor_b_a
            amor_d = self.amor_d_a
            amor_diag1 = self.amor_diag1_a
            amor_diag2 = self.amor_diag2_a
            triu_mask = getattr(self, "triu_mask_a")
            diag_idx = getattr(self, "diag_idx_aux")

        else:
            if self.flow_type != "t-sylvester":
                amor_q = self.amor_q
            last = self.z_dim_last
            amor_b = self.amor_b
            amor_d = self.amor_d
            amor_diag1 = self.amor_diag1
            amor_diag2 = self.amor_diag2
            triu_mask = getattr(self, "triu_mask")
            diag_idx = getattr(self, "diag_idx")

        if self.flow_type == "o-sylvester" and auxiliary:
            amor_q = self.amor_q_a
            last = self.a_dim

        batch_size = x.size(0)
        if auxiliary:
            (z_q, mean_z, var_z), h = self.aux_encoder(x, y=torch.FloatTensor([]).to(self.device),
                                                       a=torch.FloatTensor([]).to(self.device),
                                                       input_shape=self.input_shape)
        else:
            # n = int(torch.prod(torch.Tensor(self.input_shape)))
            #x1 = x[:, :n]
            try:
                (z_q, mean_z, var_z), h = self.encoder(x)
            except:
                z = self.encoder(x)
                if self.flow_type == "o-sylvester":
                    mean_z, var_z = torch.mean(z, 0).view(-1, 1), torch.var(z, 0).view(-1, 1).to(self.device)
                    z1 = z
                else:
                    z1 = torch.transpose(torch.Tensor(z), 0, 1)
                    mean_z, var_z = torch.mean(z1, 0).view(-1, 1).to(self.device), torch.var(z1, 0).view(-1, 1).to(self.device)
                # self.kl_divergence = -torch.sum(self._kld(z1, q2, i=0, h_last=h))
                z_q = z1

        full_d = amor_d(z)

        diag1 = amor_diag1(z)
        diag2 = amor_diag2(z)
        # h = z.view(-1, self.h_last)
        # mean_z = self.q_z_mean(h)
        # var_z = self.q_z_var(h)

        # Amortized r1, r2, q, b for all flows

        b = amor_b(z)

        if self.flow_type == "o-sylvester":
            full_d = full_d.reshape(batch_size, self.num_elements, self.num_elements, self.n_flows)
            diag1 = diag1.reshape(batch_size, self.num_elements, self.n_flows)
            diag2 = diag2.reshape(batch_size, self.num_elements, self.n_flows)
            b = b.reshape(batch_size, 1, self.num_elements, self.n_flows)
            q = amor_q(z)
        elif self.flow_type == "h-sylvester" or self.flow_type == "t-sylvester":
            full_d = full_d.reshape(batch_size, last, last, self.n_flows)
            diag1 = diag1.reshape(batch_size, last, self.n_flows)
            diag2 = diag2.reshape(batch_size, last, self.n_flows)
            b = b.reshape(batch_size, 1, last, self.n_flows)

            if self.flow_type == "h-sylvester":
                q = amor_q[i].to(self.device)(z)
            else:
                q = None
        else:
            exit(self.flow_type + "is not implemented")

        r1 = full_d * triu_mask
        r2 = full_d.transpose(2, 1) * triu_mask

        r1[:, diag_idx, diag_idx, :] = diag1
        r2[:, diag_idx, diag_idx, :] = diag2

        return (mean_z, var_z, r1, r2, q, b), x, z_q

    def forward(self, x, y=torch.Tensor([]), a=torch.Tensor([]), k=0, auxiliary=False):
        """
        Forward pass with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """

        y = y.to(self.device)
        a = a.to(self.device)
        self.log_det_j = 0.
        log_det_jacobian = 0
        (z_mu, z_var, r1, r2, q, b), x, z_q = self.encode(x, y, a, auxiliary=auxiliary, i=k)
        self.sylvester_params = (r1, r2, q, b)
        if self.flow_type == "o-sylvester":
            q_ortho = self.batch_construct_orthogonal(q, auxiliary=auxiliary)
        elif self.flow_type == "h-sylvester":
            q_ortho = self.batch_construct_householder_orthogonal(q, auxiliary=auxiliary)
        else:
            q_ortho = None
        # Sample z_0
        z, mu, var = self.GaussianSample(z_q)
        z = [z]
        # Normalizing flows
        for i in range(self.n_flows):
            flow_k = getattr(self, 'flow_' + str(k) + "_" + str(i) + "_" + str(auxiliary))
            if self.flow_type in ["o-sylvester"]:
                z_k, log_det_jacobian = flow_k(z[i],
                                               r1[:, :, :, i],
                                               r2[:, :, :, i],
                                               q_ortho[i, :, :, :],
                                               b[:, :, :, i]
                                               )
            elif self.flow_type in ["h-sylvester"]:
                q_k = q_ortho[i]
                z_k, log_det_jacobian = flow_k(z[i],
                                               r1[:, :, :, i],
                                               r2[:, :, :, i],
                                               q_k,
                                               b[:, :, :, i]
                                               )
            elif self.flow_type in ["t-sylvester"]:
                if k % 2 == 1:
                    # Alternate with reorderering z for triangular flow
                    permute_z = self.flip_idx
                else:
                    permute_z = None
                z_k, log_det_jacobian = flow_k(z[i],
                                               r1[:, :, :, i],
                                               r2[:, :, :, i],
                                               b[:, :, :, i],
                                               permute_z,
                                               sum_ldj=True)
            else:
                exit("Non implemented")
            z.append(z_k)
            self.log_det_j += log_det_jacobian
        log_p_zk = log_standard_gaussian(z[-1])
        # ln q(z_0)  (not averaged)
        # mu, log_var, r1, r2, q, b = q_param_inverse
        log_q_z0 = log_gaussian(z[0], mu, log_var=var) - self.log_det_j
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        self.kl_divergence = log_q_z0 - log_p_zk
        x_mean = self.sample(z[-1], y)

        return x_mean, self.kl_divergence
        # return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]

    def run_sylvester(self, x, y=torch.Tensor([]), a=torch.Tensor([]), k=0, auxiliary=False, exception=False):
        """
        Forward pass with orthogonal sylvester flows for the transformation z_0 -> z_1 -> ... -> z_k.
        Log determinant is computed as log_det_j = N E_q_z0[\sum_k log |det dz_k/dz_k-1| ].
        """
        y = y.to(self.device)
        a = a.to(self.device)
        if len(x.shape) == 3:
            x = x.view(-1, self.input_shape[0], self.input_shape[1], self.input_shape[2])
        self.log_det_j = 0.
        (z_mu, z_var, r1, r2, q, b), x, z_q = self.encode(x, y, a, i=k, auxiliary=auxiliary)
        # Orthogonalize all q matrices
        if self.flow_type == "o-sylvester":
            q_ortho = self.batch_construct_orthogonal(q, auxiliary)
        elif self.flow_type == "h-sylvester":
            q_ortho = self.batch_construct_householder_orthogonal(q, auxiliary)
        else:
            q_ortho = None
        # Sample z_0

        z = [self.GaussianSample.reparameterize(z_mu, z_var)]
        # Normalizing flows
        for i in range(self.n_flows):
            flow_k = getattr(self, 'flow_' + str(k) + "_" + str(i) + "_" + str(auxiliary))
            if self.flow_type in ["o-sylvester"]:
                try:
                    z_k, log_det_jacobian = flow_k(zk=z[i], r1=r1[:, :, :, i], r2=r2[:, :, :, i],
                                                   q_ortho=q_ortho[i, :, :, :], b=b[:, :, :, i])
                except:

                    z_k, log_det_jacobian = flow_k(zk=z[:, i], r1=r1[:, :, :, i], r2=r2[:, :, :, i],
                                                   q_ortho=q_ortho[i, :, :, :], b=b[:, :, :, i])

            elif self.flow_type in ["h-sylvester"]:
                q_k = q_ortho[i]
                z_k, log_det_jacobian = flow_k(z[i], r1[:, :, :, i], r2[:, :, :, i], q_k, b[:, :, :, i])
            elif self.flow_type in ["t-sylvester"]:
                if k % 2 == 1:
                    # Alternate with reorderering z for triangular flow
                    permute_z = self.flip_idx
                else:
                    permute_z = None
                z_k, log_det_jacobian = flow_k(zk=z[i], r1=r1[:, :, :, i], r2=r2[:, :, :, i], b=b[:, :, :, i],
                                               permute_z=permute_z,
                                               sum_ldj=True, auxiliary=auxiliary)
            else:
                exit("Non implemented")
            z.append(z_k)
            self.log_det_j += log_det_jacobian
        log_p_zk = log_standard_gaussian(z[-1])
        # ln q(z_0)  (not averaged)
        # mu, log_var, r1, r2, q, b = q_param_inverse
        log_q_z0 = log_gaussian(z[0], z_mu, log_var=z_var) - self.log_det_j
        # N E_q0[ ln q(z_0) - ln p(z_k) ]
        self.kl_divergence = log_q_z0 - log_p_zk
        if auxiliary and not exception:
            x_mean = None
        else:
            # if len(y) == 0:
            x_mean = self.sample(z[-1], y)

        return x_mean, z_mu, z_var, self.log_det_j, z[0], z[-1]
