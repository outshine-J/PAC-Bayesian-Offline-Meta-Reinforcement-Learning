import math
from collections import OrderedDict
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        # For compability with Torchmeta
        self.named_meta_parameters = self.named_parameters
        self.meta_parameters = self.parameters

    def update_params(self, loss, params=None, step_size=0.5, first_order=False):
        """Apply one step of gradient descent on the loss function `loss`, with
        step-size `step_size`, and returns the updated parameters of the neural
        network.
        """
        if params is None:
            params = OrderedDict(self.named_meta_parameters())

        grads = torch.autograd.grad(loss, params.values(),
                                    create_graph=not first_order)

        updated_params = OrderedDict()
        for (name, param), grad in zip(params.items(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params


# -------------------------------------------------------------------------------------------
#  Stochastic linear layer
# -------------------------------------------------------------------------------------------
class StochasticLinear(nn.Module):
    def __init__(self, in_dim, out_dim, log_var_init=None, use_bias=True):
        super(StochasticLinear, self).__init__()

        if log_var_init is None:
            log_var_init = {'mean': -10, 'std': 0.1}

        self.in_dim = in_dim
        self.out_dim = out_dim
        weights_size = (out_dim, in_dim)
        self.use_bias = use_bias
        if use_bias:
            bias_size = out_dim
        else:
            bias_size = None
        self.create_stochastic_layer(weights_size, bias_size)
        init_stochastic_linear(self, log_var_init)
        self.eps_std = 1.0

    def create_stochastic_layer(self, weights_shape, bias_size):
        # create the layer parameters
        # values initialization is done later
        self.weights_shape = weights_shape
        self.weights_count = list_mult(weights_shape)
        if bias_size is not None:
            self.weights_count += bias_size
        self.w_mu = get_param(weights_shape)
        self.w_log_var = get_param(weights_shape)
        self.w = {'mean': self.w_mu, 'log_var': self.w_log_var}
        if bias_size is not None:
            self.b_mu = get_param(bias_size)
            self.b_log_var = get_param(bias_size)
            self.b = {'mean': self.b_mu, 'log_var': self.b_log_var}

    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def forward(self, x):

        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)
        # self.operation should be linear or conv

        if self.use_bias:
            b_var = torch.exp(self.b_log_var)
            bias_mean = self.b['mean']
        else:
            b_var = None
            bias_mean = None

        out_mean = F.linear(x, self.w['mean'], bias=bias_mean)

        eps_std = self.eps_std
        if eps_std == 0.0:
            layer_out = out_mean
        else:
            w_var = torch.exp(self.w_log_var)
            out_var = F.linear(x.pow(2), w_var, bias=b_var)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output:
            noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            # noise = eps_std * torch.randn_like(out_mean, requires_grad=False)

            out_var = F.relu(out_var) # to avoid nan due to numerical errors  TODO
            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out

    def set_eps_std(self, eps_std):
        old_eps_std = self.eps_std
        self.eps_std = eps_std
        return old_eps_std

# ---------------------------------------------------------- #
#             Auxiliary Functions                            #
# ---------------------------------------------------------- #

def list_mult(L):
    return reduce(lambda x, y: x * y, L)

def get_param(shape):
    # create a parameter
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.empty(*shape))

def init_stochastic_linear(m, log_var_init):
    n = m.w_mu.size(1)
    stdv = math.sqrt(1. / n)
    m.w_mu.data.uniform_(-stdv, stdv)
    if m.use_bias:
        m.b_mu.data.uniform_(-stdv, stdv)
        m.b_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])
    m.w_log_var.data.normal_(log_var_init['mean'], log_var_init['std'])