# Modified from https://github.com/xinntao/Real-ESRGAN/blob/HEAD/realesrgan/data/realesrgan_dataset.py

import math
import random

import cv2
import numpy as np
import torch
from litsr.data.image_folder import ImageFolder
from litsr.transforms import single2tensor, uint2single, augment
from litsr.utils.registry import DatasetRegistry
from torch.utils import data

from .real_esrgan_degrade import circular_lowpass_kernel, random_mixed_kernels


@DatasetRegistry.register()
class RealESRGANDataset(data.Dataset):
    """
    Dataset used for Real-ESRGAN model.
    """

    def __init__(
        self,
        datapath,
        scale,
        is_train,
        degradation_options,
        lr_img_sz=None,
        rgb_range=1,
        repeat=1,
        data_length=None,
        cache=None,
        first_k=None,
        mean=None,
        std=None,
        return_img_name=False,
    ):
        self.scale = scale
        self.lr_img_sz = lr_img_sz
        self.repeat = repeat or 1
        self.rgb_range = rgb_range or 1
        assert self.rgb_range == 1
        self.is_train = is_train
        assert not mean and not std
        self.return_img_name = return_img_name
        self.dataset = ImageFolder(
            datapath,
            repeat=self.repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.file_names = self.dataset.filenames
        self.opt = degradation_options

        # blur settings for the first degradation
        self.blur_kernel_size = self.opt["blur_kernel_size"]
        self.kernel_list = self.opt["kernel_list"]
        self.kernel_prob = self.opt["kernel_prob"]
        self.blur_sigma = self.opt["blur_sigma"]
        self.betag_range = self.opt["betag_range"]
        self.betap_range = self.opt["betap_range"]
        self.sinc_prob = self.opt["sinc_prob"]

        # blur settings for the second degradation
        self.blur_kernel_size2 = self.opt["blur_kernel_size2"]
        self.kernel_list2 = self.opt["kernel_list2"]
        self.kernel_prob2 = self.opt["kernel_prob2"]
        self.blur_sigma2 = self.opt["blur_sigma2"]
        self.betag_range2 = self.opt["betag_range2"]
        self.betap_range2 = self.opt["betap_range2"]
        self.sinc_prob2 = self.opt["sinc_prob2"]

        # a final sinc filter
        self.final_sinc_prob = self.opt["final_sinc_prob"]

        self.kernel_range = [
            2 * v + 1 for v in range(3, 11)
        ]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(
            21, 21
        ).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1

    def __getitem__(self, idx):
        img_gt = self.dataset[idx]  # Uint8
        img_gt = uint2single(img_gt)

        # -------------------- augmentation for training: flip, rotation -------------------- #
        img_gt = augment(img_gt, self.opt["use_hflip"], self.opt["use_rot"])

        # crop or pad to 400: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        crop_pad_size = 400
        # pad
        if h < crop_pad_size or w < crop_pad_size:
            pad_h = max(0, crop_pad_size - h)
            pad_w = max(0, crop_pad_size - w)
            img_gt = cv2.copyMakeBorder(
                img_gt, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        # crop
        if img_gt.shape[0] > crop_pad_size or img_gt.shape[1] > crop_pad_size:
            h, w = img_gt.shape[0:2]
            # randomly choose top and left coordinates
            top = random.randint(0, h - crop_pad_size)
            left = random.randint(0, w - crop_pad_size)
            img_gt = img_gt[top : top + crop_pad_size, left : left + crop_pad_size, ...]

        # Generate kernels (used in the first degradation)
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob"]:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma,
                [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # Generate kernels (used in the second degradation)
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt["sinc_prob2"]:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2,
                [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # sinc kernel
        if np.random.uniform() < self.opt["final_sinc_prob"]:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = single2tensor(img_gt)
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return_d = {
            "gt": img_gt,
            "kernel1": kernel,
            "kernel2": kernel2,
            "sinc_kernel": sinc_kernel,
        }
        return return_d

    def __len__(self):
        return len(self.dataset)
