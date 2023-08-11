import argparse
import os

import numpy as np
import torch
from litsr.data import PairedImageDataset, SingleImageDataset, DownsampledDataset
from litsr.metrics import calc_fid
from litsr.utils import mkdirs, read_yaml
from matplotlib import pyplot as plt
from pytorch_lightning import seed_everything
from tqdm import tqdm
import random

from models import load_model

seed_everything(123)


def make_dataloaders(scale, config):
    dataset = PairedImageDataset(
        lr_path="load/WorldStrat_png/test/LR",
        hr_path="load/WorldStrat_png/test/HR",
        scale=scale,
        is_train=False,
        cache="bin",
        rgb_range=config.data_module.args.get("rgb_range", 1),
        mean=config.data_module.args.get("mean"),
        std=config.data_module.args.get("std"),
        return_img_name=True,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    return loader


def test(args):
    # setup device
    if args.random_seed:
        seed_everything(random.randint(0, 1000))
    device = (
        torch.device("cuda", index=int(args.gpu)) if args.gpu else torch.device("cpu")
    )

    exp_path = os.path.dirname(os.path.dirname(args.checkpoint))
    ckpt_path = args.checkpoint

    # read config
    config = read_yaml(os.path.join(exp_path, "hparams.yaml"))

    # create model
    model = load_model(config, ckpt_path, strict=False)
    model.to(device)
    model.eval()

    scale = config.data_module.args.scale

    dataloader = make_dataloaders(scale=scale, config=config)

    rslt_path = os.path.join(exp_path, "results", "worldstrat")

    lr_path = rslt_path.replace("results", "lr_sample")
    mkdirs([rslt_path, lr_path])

    psnrs, ssims, run_times, losses = [], [], [], []
    for batch in tqdm(dataloader, total=len(dataloader.dataset)):
        lr, hr, name = batch
        batch = (lr.to(device), hr.to(device), name)
        # do test
        with torch.no_grad():
            rslt = model.test_step_lr_hr_paired(batch)

        file_path = os.path.join(rslt_path, rslt["name"])
        lr_file_path = os.path.join(lr_path, rslt["name"])
        # print(rslt)
        # break
        if "log_img_sr" in rslt.keys():
            if isinstance(rslt["log_img_sr"], torch.Tensor):
                rslt["log_img_sr"] = rslt["log_img_sr"].cpu().numpy().transpose(1, 2, 0)
            plt.imsave(file_path, rslt["log_img_sr"])
        if "log_img_lr" in rslt.keys():
            if isinstance(rslt["log_img_lr"], torch.Tensor):
                rslt["log_img_lr"] = rslt["log_img_lr"].cpu().numpy().transpose(1, 2, 0)
            plt.imsave(lr_file_path, rslt["log_img_lr"])
        if "val_loss" in rslt.keys():
            losses.append(rslt["val_loss"])
        if "val_psnr" in rslt.keys():
            psnrs.append(rslt["val_psnr"])
        if "val_ssim" in rslt.keys():
            ssims.append(rslt["val_ssim"])
        if "time" in rslt.keys():
            run_times.append(rslt["time"])

    if losses:
        mean_loss = torch.stack(losses).mean()
        print("- Loss: {:.4f}".format(mean_loss))
    if psnrs:
        mean_psnr = np.array(psnrs).mean()
        print("- PSNR: {:.4f}".format(mean_psnr))
    if ssims:
        mean_ssim = np.array(ssims).mean()
        print("- SSIM: {:.4f}".format(mean_ssim))
    if run_times:
        mean_runtime = np.array(run_times[1:]).mean()
        print("- Runtime : {:.4f}".format(mean_runtime))

    hr_path = "load/WorldStrat_png/test/LR"
    paths = [rslt_path, hr_path]
    fid_score = calc_fid(paths)
    print("- FID : {:.4f}".format(fid_score))
    print("=" * 42)


def getTestParser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint", type=str, help="checkpoint index")
    parser.add_argument(
        "-g", "--gpu", default="0", type=str, help="indices of GPUs to enable"
    )
    parser.add_argument("--random_seed", action="store_true")

    return parser


test_parser = getTestParser()

if __name__ == "__main__":
    args = test_parser.parse_args()
    test(args)
