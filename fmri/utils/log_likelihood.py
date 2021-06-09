# Borrowed from https://github.com/jmtomczak/sylvester-flows/blob/master/utils/log_likelihood.py

from __future__ import print_function
import numpy as np
from scipy.special import logsumexp
import torch
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = "cpu"


# Function borrowed from: https://github.com/jmtomczak/sylvester-flows/blob/927bb28065552d50e08a4f5bdc584d68073c7627/utils/distributions.py
def log_normal_diag(x, mean, log_var, average=False, reduce=True):
    log_norm = -0.5 * (log_var + (x - mean) * (x - mean) * log_var.exp().reciprocal())
    if reduce:
        if average:
            return torch.mean(log_norm)
        else:
            return torch.sum(log_norm)
    else:
        return log_norm


def calculate_likelihood(dataloader, model, S=100, MB=4):

    # set auxiliary variables for number of training and test sets
    N_test = len(dataloader)

    likelihood_array = []
    # inputs = inputs.view(-1, *args.inputs_size)

    likelihood_test = []
    ys = []

    if S <= MB:
        R = 1
    else:
        R = S // MB
        S = MB

    for j, sample in enumerate(dataloader):
        audio, _, _ = sample
        if j % 1 == 0:
            print('Progress: {:.2f}%'.format(j / (1. * N_test) * 100))

        x_single = audio.to(device)

        a = []
        for r in range(0, R):
            # Repeat it for all training points
            x = x_single.expand(S, *x_single.size()[1:]).contiguous()

            y, mu, log_var, x_mean = model(x.unsqueeze(1), "last")
            x_mean = x_mean.squeeze()
            ys.append(y.clone().detach())
            a_tmp = log_normal_diag(x_mean.view(-1, 1), mu.view(-1, 1), log_var.view(-1, 1))

            a.append(-a_tmp.item())
            del x_mean, a_tmp, mu, log_var, y
        # calculate max
        a = np.asarray(a)
        a = np.reshape(a, (a.shape[0], 1))
        likelihood_x = -logsumexp(a)
        likelihood_test.append(likelihood_x - np.log(len(a)))
        del x_single

    likelihood_test = np.array(likelihood_test)

    nll = np.mean(likelihood_test)

    return nll, likelihood_test, ys
