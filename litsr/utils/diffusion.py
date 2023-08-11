# A translation of https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/diffusion_utils_2.py
# from TensorFlow with some help from https://github.com/rosinality/denoising-diffusion-pytorch

import math

import numpy as np
import torch
from torch import nn


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


def extract(input, t, shape):
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def get_beta_schedule(schedule, start, end, n_timestep, cosine_s=8e-3):
    def _warmup_beta(start, end, n_timestep, warmup_frac):
        betas = end * torch.ones(n_timestep, dtype=torch.float64)
        warmup_time = int(n_timestep * warmup_frac)
        betas[:warmup_time] = torch.linspace(
            start, end, warmup_time, dtype=torch.float64
        )

        return betas

    if schedule == "quad":
        betas = (
            torch.linspace(start**0.5, end**0.5, n_timestep, dtype=torch.float64)
            ** 2
        )
    elif schedule == "linear":
        betas = torch.linspace(start, end, n_timestep, dtype=torch.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(start, end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(start, end, n_timestep, 0.5)
    elif schedule == "const":
        betas = end * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / (torch.linspace(n_timestep, 1, n_timestep, dtype=torch.float64))
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)

    return betas


class Diffusion(nn.Module):
    def __init__(
        self,
        beta_type,
        beta_start,
        beta_end,
        n_timestep,
        model_var_type,
        model_mean_type,
        loss_type="l2",
        t_encode_mode="discrete",  # discrete | continuous
    ):
        super().__init__()

        self.t_encode_mode = t_encode_mode
        assert self.t_encode_mode in (
            "discrete",
            "continuous",
        ), "q_sample_mode should in [discrete | continuous]"
        betas = get_beta_schedule(beta_type, beta_start, beta_end, n_timestep).type(
            torch.float64
        )

        timesteps = betas.shape[0]
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.num_timesteps = int(timesteps)

        self.model_var_type = model_var_type  # learned, fixedsmall, fixedlarge
        self.model_mean_type = model_mean_type  # xprev, xstart, eps

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas", alphas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_alphas_cumprod_prev", torch.sqrt(alphas_cumprod_prev))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            torch.log(
                torch.cat(
                    (
                        posterior_variance[1].view(1, 1),
                        posterior_variance[1:].view(-1, 1),
                    ),
                    0,
                )
            ).view(-1),
        )
        self.register(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
        )
        self.register(
            "posterior_mean_coef2",
            ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )

        self.loss_fn = {"l1": nn.L1Loss(), "l2": nn.MSELoss()}[loss_type]

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_mean_variance(self, x_0, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
        variance = extract(1.0 - self.alphas_cumprod, t, x_0.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_0.shape)
        return mean, variance, log_variance

    def q_sample(self, x_0, t, noise=None):
        """Add noise to x_0 to obtain x_t"""
        if noise is None:
            noise = torch.randn_like(x_0)

        if self.t_encode_mode == "discrete":
            x_t = (
                extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
                + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
            )
            return x_t, t

        elif self.t_encode_mode == "continuous":
            b = x_0.shape[0]
            sqrt_alpha_cumprod_t_prev = extract(self.sqrt_alphas_cumprod_prev, t, [b])
            sqrt_alpha_cumprod_t = extract(self.sqrt_alphas_cumprod, t, [b])

            continuous_sqrt_alpha_cumprod = (
                torch.rand(b, device=x_0.device)
                * (sqrt_alpha_cumprod_t_prev - sqrt_alpha_cumprod_t)
                + sqrt_alpha_cumprod_t
            ).view(b, 1, 1, 1)

            x_t = (
                continuous_sqrt_alpha_cumprod * x_0
                + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
            )
            return x_t, continuous_sqrt_alpha_cumprod

    def q_posterior_mean_variance(self, x_0, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var_clipped

    def p_mean_variance(self, model, x, t, clip_denoised, return_pred_x0, **kwargs):
        if self.t_encode_mode == "discrete":
            model_output = model(x, t, **kwargs)
        elif self.t_encode_mode == "continuous":
            b = x.shape[0]
            t_noise_level = (
                torch.FloatTensor([self.sqrt_alphas_cumprod[t]])
                .repeat(b, 1)
                .to(x.device)
            )
            model_output = model(x, t_noise_level, **kwargs)

        # Learned or fixed variance?
        if self.model_var_type == "learned":
            model_output, log_var = torch.split(model_output, 2, dim=-1)
            var = torch.exp(log_var)
        elif self.model_var_type in ["fixedsmall", "fixedlarge"]:
            # below: only log_variance is used in the KL computations
            var, log_var = {
                # for 'fixedlarge', we set the initial (log-)variance like so to get a better decoder log likelihood
                "fixedlarge": (
                    self.betas,
                    torch.log(
                        torch.cat(
                            (
                                self.posterior_variance[1].view(1, 1),
                                self.betas[1:].view(-1, 1),
                            ),
                            0,
                        )
                    ).view(-1),
                ),
                "fixedsmall": (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]

            var = extract(var, t, x.shape) * torch.ones_like(x)
            log_var = extract(log_var, t, x.shape) * torch.ones_like(x)
        else:
            raise NotImplementedError(self.model_var_type)

        # Mean parameterization
        _maybe_clip = lambda x_: (x_.clamp(min=-1, max=1) if clip_denoised else x_)

        if self.model_mean_type == "xprev":
            # the model predicts x_{t-1}
            pred_x0 = _maybe_clip(
                self.predict_start_from_prev(x_t=x, t=t, x_prev=model_output)
            )
            mean = model_output
        elif self.model_mean_type == "xstart":
            # the model predicts x_0
            pred_x0 = _maybe_clip(model_output)
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        elif self.model_mean_type == "eps":
            # the model predicts epsilon
            pred_x0 = _maybe_clip(
                self._predict_start_from_noise(x_t=x, t=t, noise=model_output)
            )
            mean, _, _ = self.q_posterior_mean_variance(x_0=pred_x0, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        if return_pred_x0:
            return mean, var, log_var, pred_x0
        else:
            return mean, var, log_var

    def _predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def _predict_noise_from_start(self, x_t, t, pred_xstart):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def p_sample(
        self,
        model,
        x,
        t,
        noise_fn,
        clip_denoised=True,
        return_pred_x0=False,
        **kwargs,
    ):
        mean, _, log_var, pred_x0 = self.p_mean_variance(
            model, x, t, clip_denoised, return_pred_x0=True, **kwargs
        )
        noise = noise_fn(x.shape, dtype=x.dtype).to(x.device)

        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape).to(x.device)
        sample = mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

        return (sample, pred_x0) if return_pred_x0 else sample

    @torch.no_grad()
    def p_sample_loop(self, model, shape, noise_fn=torch.randn, **kwargs):
        device = self.betas.device
        img = noise_fn(shape).to(device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
                return_pred_x0=False,
                **kwargs,
            )

        return img

    @torch.no_grad()
    def p_sample_loop_fast(
        self,
        model,
        shape,
        device,
        noise_fn,
        skip=10,
        eta=0.0,
        include_x0_pred_freq=10,
        **kwargs,
    ):
        seq = range(0, self.num_timesteps, skip)
        seq_next = [0] + list(seq[:-1])

        x = noise_fn(shape).to(device)

        x0_preds = []
        for idx, i, j in zip(range(len(seq)), reversed(seq), reversed(seq_next)):
            t = torch.full((shape[0],), i, dtype=torch.int64).to(device)
            t_prev = torch.full((shape[0],), j, dtype=torch.int64).to(device)

            alpha_bar = extract(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = extract(self.alphas_cumprod, t_prev, x.shape)

            _, _, _, pred_x0 = self.p_mean_variance(
                model, x, t, True, return_pred_x0=True, **kwargs
            )

            if idx % include_x0_pred_freq == 0:
                x0_preds.append(pred_x0)

            eps = self._predict_noise_from_start(x, t, pred_x0)

            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar))
                * torch.sqrt(1 - alpha_bar / alpha_bar_prev)
            )

            noise = torch.randn_like(x)

            mean_pred = (
                pred_x0 * torch.sqrt(alpha_bar_prev)
                + torch.sqrt((1 - alpha_bar_prev) - sigma**2) * eps
            )

            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0

            x = mean_pred + nonzero_mask * sigma * noise

        return x, torch.cat(x0_preds, dim=0)

    @torch.no_grad()
    def p_sample_loop_fastdpm_var(
        self,
        model,
        shape,
        device,
        noise_fn,
        num_steps,
        schedule="quadratic",
        eta=0.0,
        **kwargs,
    ):
        """
        Fast sampling method from FastDPM: On Fast Sampling of Diffusion Probabilistic Models

        schedule(str): linear | quadratic
        """

        from .fastdpm_utils import calc_diffusion_hyperparams
        from .fastdpm_utils import get_VAR_noise

        diffusion_config = {
            "beta_0": self.beta_start,
            "beta_T": self.beta_end,
            "T": self.num_timesteps,
        }
        _dh = calc_diffusion_hyperparams(**diffusion_config)
        T, Alpha, Alpha_bar, Beta, kappa = (
            _dh["T"],
            _dh["Alpha"],
            _dh["Alpha_bar"],
            _dh["Beta"],
            eta,
        )

        assert len(Alpha_bar) == T
        assert len(shape) == 4
        assert 0.0 <= kappa <= 1.0

        user_defined_eta = get_VAR_noise(diffusion_config, num_steps, schedule)
        # compute diffusion hyperparameters for user defined noise
        T_user = len(user_defined_eta)
        Beta_tilde = torch.from_numpy(user_defined_eta).to(torch.float32).to(device)
        Gamma_bar = 1 - Beta_tilde
        for t in range(1, T_user):
            Gamma_bar[t] *= Gamma_bar[t - 1]

        assert Gamma_bar[0] <= Alpha_bar[0] and Gamma_bar[-1] >= Alpha_bar[-1]

        x = noise_fn(shape, dtype=torch.float32).to(device)
        # continuous_steps = _precompute_VAR_steps(_dh, user_defined_eta)
        with torch.no_grad():
            for i, t_noise_level in enumerate(reversed(Gamma_bar)):
                t_noise_level = (
                    torch.FloatTensor([torch.sqrt(t_noise_level)])
                    .repeat(shape[0], 1)
                    .to(x.device)  # 这里加了sqrt
                )
                pred_x0 = model(
                    x,
                    t_noise_level,
                    ctx=kwargs.get("ctx"),
                    k_v=kwargs.get("k_v"),
                )
                pred_x0 = pred_x0.clamp(min=-1, max=1)
                # _, _, _, pred_x0 = self.p_mean_variance(
                #     model, x, t_noise_level, True, return_pred_x0=True, **kwargs
                # )
                if i == T_user - 1:  # the next step is to generate x_0
                    alpha_next = torch.tensor(1.0)
                    sigma = torch.tensor(0.0)
                else:
                    alpha_next = Gamma_bar[T_user - 1 - i - 1]
                    sigma = kappa * torch.sqrt(
                        (1 - alpha_next)
                        / (1 - Gamma_bar[T_user - 1 - i])
                        * (1 - Gamma_bar[T_user - 1 - i] / alpha_next)
                    )

                epsilon_theta = (
                    x - torch.sqrt(Gamma_bar[T_user - 1 - i]) * pred_x0
                ) / torch.sqrt(1 - Gamma_bar[T_user - 1 - i])

                x *= torch.sqrt(alpha_next / Gamma_bar[T_user - 1 - i])
                c = torch.sqrt(1 - alpha_next - sigma**2) - torch.sqrt(
                    1 - Gamma_bar[T_user - 1 - i]
                ) * torch.sqrt(alpha_next / Gamma_bar[T_user - 1 - i])
                noise = noise_fn(shape, dtype=torch.float32).to(device)
                x += c * epsilon_theta + sigma * noise

        return x

    @torch.no_grad()
    def p_sample_loop_progressive(
        self,
        model,
        shape,
        device,
        noise_fn=torch.randn,
        include_x0_pred_freq=50,
        **kwargs,
    ):
        img = noise_fn(shape, dtype=torch.float32).to(device)

        num_recorded_x0_pred = self.num_timesteps // include_x0_pred_freq
        x0_preds_ = torch.zeros(
            (shape[0], num_recorded_x0_pred, *shape[1:]), dtype=torch.float32
        ).to(device)

        for i in reversed(range(self.num_timesteps)):
            # Sample p(x_{t-1} | x_t) as usual
            img, pred_x0 = self.p_sample(
                model=model,
                x=img,
                t=torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
                return_pred_x0=True,
                **kwargs,
            )

            # Keep track of prediction of x0
            insert_mask = np.floor(i // include_x0_pred_freq) == torch.arange(
                num_recorded_x0_pred, dtype=torch.int32, device=device
            )

            insert_mask = insert_mask.to(torch.float32).view(
                1, num_recorded_x0_pred, *([1] * len(shape[1:]))
            )
            x0_preds_ = (
                insert_mask * pred_x0[:, None, ...] + (1.0 - insert_mask) * x0_preds_
            )

        return img, x0_preds_

    def get_loss(self, model, x_0, t, noise=None, return_pred_x0=False, **kwargs):
        if noise is None:
            noise = torch.randn_like(x_0)

        x_t, t_encoded = self.q_sample(x_0=x_0, t=t, noise=noise)
        target = {
            "xprev": self.q_posterior_mean_variance(x_0=x_0, x_t=x_t, t=t)[0],
            "xstart": x_0,
            "eps": noise,
        }[self.model_mean_type]

        model_output = model(x_t, t_encoded, **kwargs)

        if return_pred_x0:
            _maybe_clip = lambda x_: x_.clamp(min=-1, max=1)

            if self.model_mean_type == "xprev":
                pred_x0 = _maybe_clip(
                    self.predict_start_from_prev(x_t=x_t, t=t, x_prev=model_output)
                )
            elif self.model_mean_type == "xstart":
                pred_x0 = _maybe_clip(model_output)
            elif self.model_mean_type == "eps":
                pred_x0 = _maybe_clip(
                    self._predict_start_from_noise(x_t=x_t, t=t, noise=model_output)
                )
            else:
                raise NotImplementedError(self.model_mean_type)
        loss = self.loss_fn(target, model_output)

        return (loss, pred_x0) if return_pred_x0 else loss
