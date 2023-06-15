import math

import torch
from torch.autograd import Variable

from src.models.policy import StochasticLinear

def get_task_complexity(prm, prior_model, post_model, n_samples, n_tasks=1, avg_empiric_loss=0, hyper_dvrg=0, dvrg=None, noised_prior=True):

    delta = prm.delta
    if not dvrg:
        dvrg = get_net_densities_divergence(prior_model, post_model, prm, noised_prior)

    s_wc = torch.tensor(prm.wc, dtype=torch.float)
    complex_term = (s_wc * dvrg) / (n_samples)

    # complex_term = torch.sqrt((1 / (2 * (n_samples - 1))) * (hyper_dvrg + dvrg + math.log(n_samples / delta)))
    # complex_term = torch.sqrt(
    #     (hyper_dvrg + dvrg + math.log(2 * n_samples * n_tasks / delta)) / (2 * (n_samples - 1)))  # corrected
    # complex_term = torch.sqrt(
    #     (hyper_dvrg + dvrg + math.log(2 * n_samples * n_tasks / delta)) / (2 * (n_samples - 1)))  # corrected

    return complex_term

def get_net_densities_divergence(prior_model, post_model, prm, noised_prior):

    prior_layers_list = [layer for layer in prior_model.children() if isinstance(layer, StochasticLinear)]
    post_layers_list = [layer for layer in post_model.children() if isinstance(layer, StochasticLinear)]

    total_dvrg = 0

    for i_layer, prior_layer in enumerate(prior_layers_list):
        post_layer = post_layers_list[i_layer]
        if hasattr(prior_layer, 'w'):
            total_dvrg += get_dvrg_element(post_layer.w, prior_layer.w, prm, noised_prior)
        if hasattr(prior_layer, 'b'):
            total_dvrg += get_dvrg_element(post_layer.b, prior_layer.b, prm, noised_prior)

    return total_dvrg

def get_dvrg_element(post, prior, prm, noised_prior=False):

    if noised_prior and prm.kappa_post > 0:
        prior_log_var = add_noise(prior['log_var'], prm.kappa_post)
        prior_mean = add_noise(prior['mean'], prm.kappa_post)
    else:
        prior_log_var = prior['log_var']
        prior_mean = prior['mean']

    post_var = torch.exp(post['log_var'])
    prior_var = torch.exp(prior_log_var)
    post_std = torch.exp(0.5 * post['log_var'])
    prior_std = torch.exp(0.5 * prior_log_var)

    numerator = (post['mean'] - prior_mean).pow(2) + post_var
    denominator = prior_var
    div_elem = 0.5 * torch.sum(prior_log_var - post['log_var'] + numerator / denominator - 1)

    assert div_elem >= 0
    return div_elem

def add_noise(param, std):
    return param + Variable(param.data.new(param.size()).normal_(0, std), requires_grad=False)


def get_meta_complexity_term(prm, hyper_kl, n_samples, n_train_tasks):
    delta = prm.delta

    s_wc = torch.tensor(prm.wc, dtype=torch.float)
    d_wc = torch.tensor(prm.wc, dtype=torch.float)
    # New
    meta_complex_term = ((s_wc + n_samples * d_wc) * (hyper_kl - math.log(delta))) / (n_samples * n_train_tasks) + math.pow(prm.bound, 2) / 4

    # meta_complex_term = torch.sqrt(
        # (hyper_kl + math.log(2 * n_train_tasks / delta)) / (2 * (n_train_tasks - 1)))  # corrected

    return meta_complex_term

def get_hyper_divergnce(prm, prior_model):

    kappa_post = prm.kappa_post
    kappa_prior = prm.kappa_prior

    norm_sqr = net_weights_magnitude(prior_model, prm, p=2)
    hyper_dvrg = (norm_sqr + kappa_post**2) / (2 * kappa_prior**2) + math.log(kappa_prior / kappa_post) - 1 / 2

    assert hyper_dvrg >= 0
    return hyper_dvrg

def net_weights_magnitude(model, prm, p=2):   # corrected
    ''' Calculates the total p-norm of the weights  |W|_p^p
        If exp_on_logs flag is on, then parameters with log_var in their name are exponented'''
    total_mag = torch.zeros(1, device=prm.device, requires_grad=True)[0]
    for (param_name, param) in model.named_parameters():
        total_mag = total_mag + param.pow(p).sum()
    return total_mag


def adjust_learning_rate_schedule(optimizer, epoch, initial_lr, decay_factor, decay_epochs):
    """The learning rate is decayed by decay_factor at each interval start """

    # Find the index of the current interval:
    interval_index = len([mark for mark in decay_epochs if mark < epoch])

    lr = initial_lr * (decay_factor ** interval_index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr