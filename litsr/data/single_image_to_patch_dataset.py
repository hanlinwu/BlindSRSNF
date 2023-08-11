import torch.utils.data as data
from litsr import transforms
from litsr.data.image_folder import ImageFolder
from litsr.utils.registry import DatasetRegistry
import torch as th


@DatasetRegistry.register()
class SingleImageToPatchDataset(data.Dataset):
    def __init__(
        self,
        img_path,
        is_train,
        patch_size=None,
        rgb_range=1,
        repeat=1,
        data_length=None,
        cache=None,
        first_k=None,
        patch_num=2,
        mean=None,
        std=None,
        return_img_name=False,
    ):
        assert not is_train ^ bool(
            patch_size
        ), "If is_train = True, the patch_size should be specified."

        self.is_train = is_train
        self.patch_size = patch_size
        self.dataset = ImageFolder(
            img_path,
            repeat=repeat,
            cache=cache,
            first_k=first_k,
            data_length=data_length,
        )
        self.patch_num = patch_num
        self.repeat = repeat
        self.rgb_range = rgb_range
        self.mean = mean
        self.std = std
        self.return_img_name = return_img_name
        self.file_names = self.dataset.filenames

    def get_patchs(self, hr):
        out = []
        hr = transforms.augment(hr)
        # extract two patches from each image
        for _ in range(self.patch_num):
            hr_patch = transforms.random_crop(hr, patch_size=self.patch_size)
            out.append(hr_patch)
        return out

    def __getitem__(self, idx):
        img = self.dataset[idx]
        if self.is_train:
            patchs = self.get_patchs(img)
        else:
            patchs = [img]

        patchs = [transforms.uint2single(p) for p in patchs]
        patchs = [transforms.single2tensor(p) * self.rgb_range for p in patchs]

        if self.mean and self.std:
            patchs = [
                transforms.normalize(p, self.mean, self.std, inplace=True)
                for p in patchs
            ]

        out = th.stack(patchs, 0)
        if self.return_img_name:
            file_name = self.file_names[idx % (len(self.dataset) // self.repeat)]
            return out, file_name
        else:
            return out

    def __len__(self):
        return len(self.dataset)
