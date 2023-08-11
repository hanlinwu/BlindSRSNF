import numpy as np
import torch.utils.data as data
from litsr import transforms
from litsr.data.image_folder import ImageFolder
from litsr.utils.registry import DatasetRegistry
from torchvision import transforms as tv_transforms
from torchvision.transforms import functional as TF
from litsr.data.bsrgan_degrade import degradation_bsrgan
from litsr.transforms import uint2single, single2tensor


@DatasetRegistry.register()
class BSRGANDataset(data.Dataset):
    def __init__(
        self,
        datapath,
        scale,
        is_train,
        lr_img_sz=None,
        rgb_range=1,
        repeat=1,
        cache=None,
        first_k=None,
        data_length=None,
        mean=None,
        std=None,
        downsample_mode="bicubic",
        return_img_name=False,
    ):
        assert not is_train ^ bool(lr_img_sz)

        self.scale = scale
        self.lr_img_sz = lr_img_sz
        self.repeat = repeat or 1
        self.rgb_range = rgb_range or 1
        self.is_train = is_train
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.downsample_mode = downsample_mode or "bicubic"
        self.dataset = ImageFolder(
            datapath,
            repeat=self.repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.file_names = self.dataset.filenames

    def __getitem__(self, idx):
        hr = self.dataset[idx]
        if self.is_train:
            lr, hr = self._transform_train(hr)
        else:
            lr, hr = self._transform_test(hr)

        if self.mean and self.std:
            transforms.normalize(lr, self.mean, self.std, inplace=True)
            transforms.normalize(hr, self.mean, self.std, inplace=True)

        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            return lr, hr, file_name
        else:
            return lr, hr

    def _transform_train(self, hr):
        H_size = 320

        augment = tv_transforms.Compose(
            [
                tv_transforms.ToPILImage(),
                tv_transforms.RandomCrop(H_size),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.RandomVerticalFlip(),
                lambda x: TF.rotate(x, 90) if np.random.random() > 0.5 else x,
            ]
        )
        hr = np.array(augment(hr))
        hr = uint2single(hr)
        lr, hr = degradation_bsrgan(hr, self.scale, self.lr_img_sz, isp_model=None)
        lr, hr = single2tensor(lr), single2tensor(hr)
        return lr, hr

    def _transform_test(self, hr):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)
