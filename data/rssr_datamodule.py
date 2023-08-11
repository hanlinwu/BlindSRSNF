from typing import Any, Optional

from litsr.utils.registry import DataModuleRegistry
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from litsr.data.downsampled_dataset import DownsampledDataset
from litsr.data.paired_image_dataset import PairedImageDataset


@DataModuleRegistry.register()
class RSSRDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train_dataset = DownsampledDataset(
                datapath=self.opt.train.data_path,
                scale=self.opt.scale,
                is_train=True,
                repeat=self.opt.train.get("data_repeat"),
                lr_img_sz=self.opt.train.lr_img_sz,
                rgb_range=self.opt.get("rgb_range"),
                cache=self.opt.train.get("data_cache"),
                first_k=self.opt.train.get("data_first_k"),
                mean=self.opt.get("mean"),
                std=self.opt.get("std"),
                downsample_mode=self.opt.train.get("downsample_mode"),
                return_img_name=False,
            )

        self.val_dataset = DownsampledDataset(
            datapath=self.opt.valid.data_path,
            scale=self.opt.scale,
            is_train=False,
            first_k=self.opt.valid.get("data_first_k"),
            rgb_range=self.opt.get("rgb_range"),
            mean=self.opt.get("mean"),
            std=self.opt.get("std"),
            return_img_name=True,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.opt.train.batch_size,
            num_workers=self.opt.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.opt.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=1,
            num_workers=self.opt.num_workers,
            pin_memory=True,
        )
