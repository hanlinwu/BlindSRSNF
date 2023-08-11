from typing import Optional

from litsr.utils.registry import DataModuleRegistry
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from litsr.data import RealESRGANDataset, SingleImageDataset


@DataModuleRegistry.register()
class RealESRGANDataModule(LightningDataModule):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

    def setup(self, stage: Optional[str]):
        if stage == "fit":
            self.train_dataset = RealESRGANDataset(
                datapath=self.opt.train.data_path,
                scale=self.opt.scale,
                is_train=True,
                degradation_options=self.opt.train.degradations,
                repeat=self.opt.train.get("data_repeat"),
                data_length=self.opt.train.get("data_length"),
                lr_img_sz=self.opt.train.lr_img_sz,
                rgb_range=self.opt.rgb_range,
                cache=self.opt.train.get("data_cache"),
                first_k=self.opt.train.get("data_first_k"),
                mean=self.opt.get("mean"),
                std=self.opt.get("std"),
                return_img_name=False,
            )

        self.val_dataset = SingleImageDataset(
            img_path=self.opt.valid.hr_data_path,
            rgb_range=self.opt.rgb_range,
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
