import functools
import math
import time
from functools import partial
from inspect import isfunction

import numpy as np
import pytorch_lightning as pl
import torch
import torch as th
from litsr.data.srmd_degrade import SRMDPreprocessing
from litsr.metrics import calc_psnr_ssim
from litsr.transforms import denormalize, normalize, tensor2uint8
from torch import nn
from torch.nn import init
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

from archs.sr3_vanilla_unet_arch import SR3VanillaUNet as UNet
from models import ModelRegistry

####################
# SR3 Diffusion
####################


def extract(input, t, shape):
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64
    )
    return betas


def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "quad":
        betas = (
            np.linspace(
                linear_start**0.5, linear_end**0.5, n_timestep, dtype=np.float64
            )
            ** 2
        )
    elif schedule == "linear":
        betas = np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)
    elif schedule == "warmup10":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.1)
    elif schedule == "warmup50":
        betas = _warmup_beta(linear_start, linear_end, n_timestep, 0.5)
    elif schedule == "const":
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(n_timestep, 1, n_timestep, dtype=np.float64)
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


# gaussian diffusion trainer class


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        image_size,
        channels=3,
        loss_type="l1",
        conditional=True,
        schedule_opt=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.set_loss()
        if schedule_opt is not None:
            self.set_new_noise_schedule(schedule_opt)

    def set_loss(self):
        if self.loss_type == "l1":
            self.loss_func = nn.L1Loss(reduction="sum")
        elif self.loss_type == "l2":
            self.loss_func = nn.MSELoss(reduction="sum")
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt):
        to_torch = partial(torch.tensor, dtype=torch.float32)

        betas = make_beta_schedule(
            schedule=schedule_opt["schedule"],
            n_timestep=schedule_opt["n_timestep"],
            linear_start=schedule_opt["linear_start"],
            linear_end=schedule_opt["linear_end"],
        )
        betas = (
            betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        )
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1.0, alphas_cumprod))

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            self.sqrt_recip_alphas_cumprod[t] * x_t
            - self.sqrt_recipm1_alphas_cumprod[t] * noise
        )

    def _predict_noise_from_start(self, x_t, t, pred_xstart):
        return (self.sqrt_recip_alphas_cumprod[t] * x_t - pred_xstart) / (
            self.sqrt_recipm1_alphas_cumprod[t]
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        )
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = (
            torch.FloatTensor([self.sqrt_alphas_cumprod_prev[t + 1]])
            .repeat(batch_size, 1)
            .to(x.device)
        )
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x,
                t=t,
                noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level),
            )
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level)
            )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        model_mean, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_log_variance

    def p_mean_variance_fast_loop(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = (
            torch.FloatTensor([self.sqrt_alphas_cumprod[t]])
            .repeat(batch_size, 1)
            .to(x.device)
        )
        x_recon = self.predict_start_from_noise(
            x,
            t=t,
            noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level),
        )

        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)

        # model_mean, posterior_log_variance = self.q_posterior(
        #     x_start=x_recon, x_t=x, t=t
        # )
        return x_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x
        )
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = 1 | (self.num_timesteps // 10)
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                img = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            for i in tqdm(
                reversed(range(0, self.num_timesteps)),
                desc="sampling loop time step",
                total=self.num_timesteps,
            ):
                img = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
        if continous:
            return ret_img
        else:
            return ret_img[-1]

    def p_sample_loop_fast(
        self,
        x_in,
        skip=50,
        eta=1.0,
        **kwargs,
    ):
        shape = x_in.shape
        device = self.betas.device

        seq = range(0, self.num_timesteps, skip)
        seq_next = [0] + list(seq[:-1])

        x = torch.randn(shape).to(device)

        for idx, t, t_prev in zip(range(len(seq)), reversed(seq), reversed(seq_next)):
            alpha_bar = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t_prev]

            pred_x0 = self.p_mean_variance_fast_loop(x, t, True, condition_x=x_in)

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

            t = torch.full((shape[0],), t, dtype=torch.int64).to(device)
            nonzero_mask = (
                (t != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
            )  # no noise when t == 0

            x = mean_pred + nonzero_mask * sigma * noise
        return x

    @torch.no_grad()
    def sample(self, batch_size=1, continous=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop(
            (batch_size, channels, image_size, image_size), continous
        )

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop_fast(x_in, skip=50, eta=1.0)  # >=20
        return self.p_sample_loop(x_in, continous)

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        # random gama
        return (
            continuous_sqrt_alpha_cumprod * x_start
            + (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in["HR"]
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t - 1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b,
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(
                -1, 1, 1, 1
            ),
            noise=noise,
        )

        if not self.conditional:
            x_recon = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            x_recon = self.denoise_fn(
                torch.cat([x_in["SR"], x_noisy], dim=1), continuous_sqrt_alpha_cumprod
            )

        loss = self.loss_func(noise, x_recon)
        return loss

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)


####################
# Lightning Model
####################


def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.normal_(m.weight.data, 1.0, std)  # BN also uses norm
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("Linear") != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find("BatchNorm2d") != -1:
        init.constant_(m.weight.data, 1.0)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type="kaiming", scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print("Initialization method [{:s}]".format(init_type))
    if init_type == "normal":
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == "kaiming":
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == "orthogonal":
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError(
            "initialization method [{:s}] not implemented".format(init_type)
        )


@ModelRegistry.register()
class SR3VanillaModel(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()

        # save hyperparameters
        self.save_hyperparameters(opt)
        self.opt = opt.lit_model.args

        # init super-resolution network
        self.model = UNet(
            in_channel=self.opt.unet.in_channel,
            out_channel=self.opt.unet.out_channel,
            norm_groups=self.opt.unet.norm_groups,
            inner_channel=self.opt.unet.inner_channel,
            channel_mults=self.opt.unet.channel_multiplier,
            attn_res=self.opt.unet.attn_res,
            res_blocks=self.opt.unet.res_blocks,
            dropout=self.opt.unet.dropout,
            image_size=self.opt.diffusion.image_size,
        )
        self.netG = GaussianDiffusion(
            self.model,
            image_size=self.opt.diffusion.image_size,
            channels=self.opt.diffusion.channels,
            loss_type=self.opt.diffusion.loss_type,
            conditional=self.opt.diffusion.conditional,
            schedule_opt=self.opt.beta_schedule,
        )
        init_weights(self.netG, init_type="orthogonal")

        self.mean, self.std = self.opt.mean, self.opt.std
        self.rgb_range = self.hparams.data_module.args.rgb_range
        self.normalize = lambda tensor: normalize(
            tensor, self.mean, self.std, inplace=True
        )
        self.denormalize = lambda tensor: denormalize(
            tensor, self.mean, self.std, inplace=True
        )

        self.degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.blur_kernel,
            blur_type=self.opt.blur_type,
            sig_min=self.opt.sig_min,
            sig_max=self.opt.sig_max,
            lambda_min=self.opt.lambda_min,
            lambda_max=self.opt.lambda_max,
            noise=self.opt.noise,
        )
        self.valid_degrade = SRMDPreprocessing(
            self.opt.scale,
            kernel_size=self.opt.valid.blur_kernel,
            blur_type=self.opt.valid.blur_type,
            sig=self.opt.valid.get("sig"),
            lambda_1=self.opt.valid.get("lambda_1"),
            lambda_2=self.opt.valid.get("lambda_2"),
            theta=self.opt.valid.get("theta"),
            noise=self.opt.valid.get("noise"),
        )

    def training_step(self, batch, *args):
        hr = batch
        hr.mul_(255.0)
        lr, _ = self.degrade(hr)
        hr.div_(255.0)
        lr.div_(255.0)
        self.normalize(hr)
        self.normalize(lr)

        lr, hr = lr[:, 0, ...], hr[:, 0, ...]

        b, c, h, w = hr.shape
        lr_upsampled = TF.resize(
            lr,
            size=(h, w),
            interpolation=InterpolationMode.BICUBIC,
        )

        l_pix = self.netG({"HR": hr, "SR": lr_upsampled})
        loss_elbo = l_pix.sum() / int(b * c * h * w)
        self.log("train/loss_elbo", loss_elbo)
        return loss_elbo

    def validation_step(self, batch, *args, **kwargs):
        # choose use ema or not

        hr, name = batch
        if len(hr.shape) == 4:
            hr = hr.unsqueeze(1)

        b, _, c, h, w = hr.shape

        hr.mul_(255.0)
        lr, _ = self.valid_degrade(hr, random=False)
        hr.div_(255.0)
        lr.div_(255.0)
        self.normalize(hr)
        self.normalize(lr)
        lr, hr = lr.squeeze(1), hr.squeeze(1)

        th.cuda.synchronize()
        tic = time.time()
        lr_upsampled = TF.resize(
            lr,
            size=(h, w),
            interpolation=InterpolationMode.BICUBIC,
        )

        sr = self.netG.super_resolution(lr_upsampled)
        if len(sr.shape) == 3:
            sr.unsqueeze_(0)

        th.cuda.synchronize()
        toc = time.time()

        self.denormalize(sr)
        self.denormalize(hr)

        crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))
        # print(sr.cpu()[0].shape, hr.cpu()[0].shape)
        sr_np, hr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0]], self.hparams.data_module.args.rgb_range
        )
        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=self.opt.valid.test_Y
        )
        return {
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "name": name[0],
            "time": toc - tic,
        }

    def validation_epoch_end(self, outputs):
        # avg_val_loss = th.stack([x["val_loss"] for x in outputs]).mean()
        avg_val_psnr = np.array([x["val_psnr"] for x in outputs]).mean()
        # self.log("val/loss", avg_val_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/psnr", avg_val_psnr, on_epoch=True, prog_bar=True, logger=True)

        log_img_sr = outputs[0]["log_img_sr"]

        self.logger.experiment.add_image(
            "img_sr", log_img_sr, self.global_step, dataformats="HWC"
        )

        return

    def test_step(self, batch, *args, **kwargs):
        return self.validation_step(batch, *args, **kwargs)

    def test_step_lr_hr_paired(self, batch, *args, **kwargs):
        # choose use ema or not

        lr, hr, name = batch
        b, c, h, w = hr.shape

        self.normalize(hr)
        self.normalize(lr)

        th.cuda.synchronize()
        tic = time.time()
        lr_upsampled = TF.resize(
            lr,
            size=(h, w),
            interpolation=InterpolationMode.BICUBIC,
        )

        sr = self.netG.super_resolution(lr_upsampled)
        if len(sr.shape) == 3:
            sr.unsqueeze_(0)

        th.cuda.synchronize()
        toc = time.time()

        self.denormalize(sr)
        self.denormalize(hr)

        crop_border = int(np.ceil(float(hr.shape[2]) / lr.shape[2]))
        # print(sr.cpu()[0].shape, hr.cpu()[0].shape)
        sr_np, hr_np = tensor2uint8(
            [sr.cpu()[0], hr.cpu()[0]], self.hparams.data_module.args.rgb_range
        )
        psnr, ssim = calc_psnr_ssim(
            sr_np, hr_np, crop_border=crop_border, test_Y=self.opt.valid.test_Y
        )
        return {
            "val_psnr": psnr,
            "val_ssim": ssim,
            "log_img_sr": sr_np,
            "name": name[0],
            "time": toc - tic,
        }

    def test_step_lr_only(self, batch, *args, **kwargs):
        # choose use ema or not

        lr, name = batch
        b, c, lr_h, lr_w = lr.shape
        h = lr_h * 4
        w = lr_w * 4

        self.normalize(lr)

        th.cuda.synchronize()
        tic = time.time()
        lr_upsampled = TF.resize(
            lr,
            size=(h, w),
            interpolation=InterpolationMode.BICUBIC,
        )

        sr = self.netG.super_resolution(lr_upsampled)
        if len(sr.shape) == 3:
            sr.unsqueeze_(0)

        th.cuda.synchronize()
        toc = time.time()

        self.denormalize(sr)

        sr_np = tensor2uint8([sr.cpu()[0]], self.hparams.data_module.args.rgb_range)
        return {
            "log_img_sr": sr_np[0],
            "name": name[0],
            "time": toc - tic,
        }

    def configure_optimizers(self):
        betas = self.opt.optimizer.get("betas") or (0.9, 0.999)
        optimizer = th.optim.Adam(
            self.parameters(), lr=self.opt.optimizer.lr, betas=betas
        )
        if self.opt.optimizer.get("lr_scheduler_step"):
            LR_scheduler = th.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.opt.optimizer.lr_scheduler_step,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        elif self.opt.optimizer.get("lr_scheduler_milestones"):
            LR_scheduler = th.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.opt.optimizer.lr_scheduler_milestones,
                gamma=self.opt.optimizer.lr_scheduler_gamma,
            )
        else:
            raise Exception("No lr settings found! ")
        return [optimizer], [LR_scheduler]
