import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torchvision.utils import save_image, make_grid

def _map_gpu(gpu):
    if gpu == 'cuda':
        return lambda x: x.cuda()
    else:
        return lambda x: x.to(torch.device('cuda:'+gpu))
    
map_gpu = _map_gpu("cuda")


def rescale(X, batch=True):
    if not batch:
        return (X - X.min()) / (X.max() - X.min())
    else:
        for i in range(X.shape[0]):
            X[i] = rescale(X[i], batch=False)
        return X


def std_normal(size):
    return map_gpu(torch.normal(0, 1, size=size))


def calc_diffusion_hyperparams(T, beta_0, beta_T):
    """
    Compute diffusion process hyperparameters

    Parameters:
    T (int):                    number of diffusion steps
    beta_0 and beta_T (float):  beta schedule start/end value, 
                                where any beta_t in the middle is linearly interpolated
    
    Returns:
    a dictionary of diffusion hyperparameters including:
        T (int), Beta/Alpha/Alpha_bar/Sigma (torch.tensor on cpu, shape=(T, ))
    """

    Beta = torch.linspace(beta_0, beta_T, T)
    Alpha = 1 - Beta
    Alpha_bar = Alpha + 0
    Beta_tilde = Beta + 0
    for t in range(1, T):
        Alpha_bar[t] *= Alpha_bar[t-1]
        Beta_tilde[t] *= (1-Alpha_bar[t-1]) / (1-Alpha_bar[t])
    Sigma = torch.sqrt(Beta_tilde)
    
    _dh = {}
    _dh["T"], _dh["Beta"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"] = T, Beta, Alpha, Alpha_bar, Sigma
    diffusion_hyperparams = _dh
    return diffusion_hyperparams


def bisearch(f, domain, target, eps=1e-8):
    """
    find smallest x such that f(x) > target

    Parameters:
    f (function):               function
    domain (tuple):             x in (left, right)
    target (float):             target value
    
    Returns:
    x (float)
    """
    # 
    sign = -1 if target < 0 else 1
    left, right = domain
    for _ in range(1000):
        x = (left + right) / 2 
        if f(x) < target:
            right = x
        elif f(x) > (1 + sign * eps) * target:
            left = x
        else:
            break
    return x


def get_VAR_noise(diffusion_config, S, schedule='linear'):
    """
    Compute VAR noise levels

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of noise levels, size = (S, )
    """
    target = np.prod(1 - np.linspace(diffusion_config["beta_0"], diffusion_config["beta_T"], diffusion_config["T"]))

    if schedule == 'linear':
        g = lambda x: np.linspace(diffusion_config["beta_0"], x, S)
        domain = (diffusion_config["beta_0"], 0.99)
    elif schedule == 'quadratic':
        g = lambda x: np.array([diffusion_config["beta_0"] * (1+i*x) ** 2 for i in range(S)])
        domain = (0.0, 0.95 / np.sqrt(diffusion_config["beta_0"]) / S)
    else:
        raise NotImplementedError

    f = lambda x: np.prod(1 - g(x))
    largest_var = bisearch(f, domain, target, eps=1e-4)
    return g(largest_var)


def get_STEP_step(diffusion_config, S, schedule='linear'):
    """
    Compute STEP steps

    Parameters:
    S (int):            approximante diffusion process length
    schedule (str):     linear or quadratic
    
    Returns:
    np array of steps, size = (S, )
    """
    if schedule == 'linear':
        c = (diffusion_config["T"] - 1.0) / (S - 1.0)
        list_tau = [np.floor(i * c) for i in range(S)]
    elif schedule == 'quadratic':
        list_tau = np.linspace(0, np.sqrt(diffusion_config["T"] * 0.8), S) ** 2
    else:
        raise NotImplementedError

    return [int(s) for s in list_tau]


def _log_gamma(x):
    # Gamma(x+1) ~= sqrt(2\pi x) * (x/e)^x  (1 + 1 / 12x)
    y = x - 1
    return np.log(2 * np.pi * y) / 2 + y * (np.log(y) - 1) + np.log(1 + 1 / (12 * y))


def _log_cont_noise(t, beta_0, beta_T, T):
    # We want log_cont_noise(t, beta_0, beta_T, T) ~= np.log(Alpha_bar[-1].numpy())
    delta_beta = (beta_T - beta_0) / (T - 1)
    _c = (1.0 - beta_0) / delta_beta
    t_1 = t + 1
    return t_1 * np.log(delta_beta) + _log_gamma(_c + 1) - _log_gamma(_c - t_1 + 1)


# Standard DDPM generation
def STD_sampling(net, size, diffusion_hyperparams):
    """
    Perform the complete sampling step according to DDPM

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors

    Returns:
    the generated images in torch.tensor, shape=size
    """

    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T
    assert len(size) == 4

    Sigma = _dh["Sigma"]

    x = std_normal(size)
    with torch.no_grad():
        for t in range(T-1, -1, -1):
            diffusion_steps = t * map_gpu(torch.ones(size[0]))
            epsilon_theta = net(x, diffusion_steps)
            x = (x - (1-Alpha[t])/torch.sqrt(1-Alpha_bar[t]) * epsilon_theta) / torch.sqrt(Alpha[t])
            if t > 0:
                x = x + Sigma[t] * std_normal(size)
    return x


# STEP
def STEP_sampling(net, size, diffusion_hyperparams, user_defined_steps, kappa):
    """
    Perform the complete sampling step according to https://arxiv.org/pdf/2010.02502.pdf
    official repo: https://github.com/ermongroup/ddim

    Parameters:
    net (torch network):            the model
    size (tuple):                   size of tensor to be generated, 
                                    usually is (number of audios to generate, channels=1, length of audio)
    diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
                                    note, the tensors need to be cuda tensors
    user_defined_steps (int list):  User defined steps (sorted)     
    kappa (float):                  factor multipled over sigma, between 0 and 1
    
    Returns:
    the generated images in torch.tensor, shape=size
    """
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, _ = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Sigma"]
    assert len(Alpha_bar) == T
    assert len(size) == 4
    assert 0.0 <= kappa <= 1.0

    T_user = len(user_defined_steps)
    user_defined_steps = sorted(list(user_defined_steps), reverse=True)

    x = std_normal(size)
    with torch.no_grad():
        for i, tau in enumerate(user_defined_steps):
            diffusion_steps = tau * map_gpu(torch.ones(size[0]))
            epsilon_theta = net(x, diffusion_steps)
            if i == T_user - 1:  # the next step is to generate x_0
                assert tau == 0
                alpha_next = torch.tensor(1.0) 
                sigma = torch.tensor(0.0) 
            else:
                alpha_next = Alpha_bar[user_defined_steps[i+1]]
                sigma = kappa * torch.sqrt((1-alpha_next) / (1-Alpha_bar[tau]) * (1 - Alpha_bar[tau] / alpha_next))
            x *= torch.sqrt(alpha_next / Alpha_bar[tau])
            c = torch.sqrt(1 - alpha_next - sigma ** 2) - torch.sqrt(1 - Alpha_bar[tau]) * torch.sqrt(alpha_next / Alpha_bar[tau])
            x += c * epsilon_theta + sigma * std_normal(size)
    return x


# VAR
def _precompute_VAR_steps(diffusion_hyperparams, user_defined_eta):
    _dh = diffusion_hyperparams
    T, Alpha, Alpha_bar, Beta = _dh["T"], _dh["Alpha"], _dh["Alpha_bar"], _dh["Beta"]
    assert len(Alpha_bar) == T

    # compute diffusion hyperparameters for user defined noise
    T_user = len(user_defined_eta)
    Beta_tilde = map_gpu(torch.from_numpy(user_defined_eta)).to(torch.float32)
    Gamma_bar = 1 - Beta_tilde
    for t in range(1, T_user):
        Gamma_bar[t] *= Gamma_bar[t-1]

    assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]
    
    continuous_steps = []
    with torch.no_grad():
        for t in range(T_user-1, -1, -1):
            t_adapted = None
            for i in range(T - 1):
                if Alpha_bar[i] >= Gamma_bar[t] > Alpha_bar[i+1]:
                    t_adapted = bisearch(f=lambda _t: _log_cont_noise(_t, Beta[0].cpu().numpy(), Beta[-1].cpu().numpy(), T), 
                                            domain=(i-0.01, i+1.01), 
                                            target=np.log(Gamma_bar[t].cpu().numpy()))
                    break
            if t_adapted is None:
                t_adapted = T - 1
            continuous_steps.append(t_adapted)  # must be decreasing
    return continuous_steps