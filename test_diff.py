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


def make_dataloaders(datasets, type, scale, config):
    dataloaders = []
    for dataset_name in datasets:
        if type == "LRHR_paired":
            dataset = PairedImageDataset(
                lr_path="load/benchmark/{0}/LR_bicubic/X{1}".format(
                    dataset_name, scale
                ),
                hr_path="load/benchmark/{0}/HR".format(dataset_name),
                scale=scale,
                is_train=False,
                cache="bin",
                rgb_range=config.data_module.args.rgb_range,
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        elif type == "LR_only":
            dataset = SingleImageDataset(
                img_path="load/benchmark/{0}".format(dataset_name),
                rgb_range=config.data_module.args.rgb_range,
                cache="bin",
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        elif type == "HR_only":
            dataset = SingleImageDataset(
                img_path="load/benchmark/{0}/HR".format(dataset_name),
                rgb_range=config.data_module.args.rgb_range,
                cache="bin",
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        elif type == "HR_downsampled":
            dataset = DownsampledDataset(
                datapath="load/benchmark/{0}/HR".format(dataset_name),
                scale=scale,
                is_train=False,
                rgb_range=config.data_module.args.rgb_range,
                cache="bin",
                mean=config.data_module.args.get("mean"),
                std=config.data_module.args.get("std"),
                return_img_name=True,
            )
        else:
            raise "Unknown dataset type"
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        dataloaders.append((dataset_name, loader))
    return dataloaders


def test(args):
    # setup device
    if args.random_seed:
        seed_everything(random.randint(0, 1000))
    device = (
        torch.device("cuda", index=int(args.gpu)) if args.gpu else torch.device("cpu")
    )

    # setup datasets
    test_datasets = [_ for _ in args.datasets.split(",")]

    exp_path = os.path.dirname(os.path.dirname(args.checkpoint))
    ckpt_path = args.checkpoint

    # read config
    config = read_yaml(os.path.join(exp_path, "hparams.yaml"))

    config.lit_model.args.valid.skip = args.skip
    config.lit_model.args.valid.eta = args.eta
    config.lit_model.args.valid.sig = args.sigma
    config.lit_model.args.valid.noise = args.noise
    config.lit_model.args.valid.lambda_1 = args.lambda1
    config.lit_model.args.valid.lambda_2 = args.lambda2
    config.lit_model.args.valid.theta = args.theta
    config.lit_model.args.valid.blur_type = config.lit_model.args.blur_type

    # create model
    model = load_model(config, ckpt_path, strict=False)
    model.to(device)
    model.eval()

    scale = config.data_module.args.scale

    dataloaders = make_dataloaders(
        datasets=test_datasets, type=args.datatype, scale=scale, config=config
    )
    for dataset_name, loader in dataloaders:
        # config result path
        rslt_path = rslt_path = os.path.join(
            exp_path,
            "results",
            dataset_name,
            "x" + str(scale),
        )
        if config.lit_model.args.blur_type == "iso_gaussian":
            rslt_path = os.path.join(
                rslt_path,
                "sig_{0}_noise_{1}".format(str(args.sigma), str(args.noise)),
            )
            print(
                "==== Dataset: {}, Scale Factor: x{:.2f}, Sigma: {:.2f}, noise: {:.2f}, eta {:.2f}, skip {:.2f}====".format(
                    dataset_name, scale, args.sigma, args.noise, args.eta, args.skip
                )
            )
        if config.lit_model.args.blur_type == "aniso_gaussian":
            rslt_path = os.path.join(
                rslt_path,
                "l1_{0}_l2_{1}_theta_{2}_noise_{3}".format(
                    str(args.lambda1),
                    str(args.lambda2),
                    str(args.theta),
                    str(args.noise),
                    str(args.eta),
                    str(args.skip),
                ),
            )
            print(
                "==== Dataset: {}, Scale Factor: x{:.2f}, Lambda: [{:.2f}, {:.2f}], theta {:.2f}, noise {:.2f}, eta {:.2f}, skip {:.2f} ====".format(
                    dataset_name,
                    scale,
                    args.lambda1,
                    args.lambda2,
                    args.theta,
                    args.noise,
                    args.eta,
                    args.skip,
                )
            )
        if args.eta != 1:
            rslt_path = rslt_path + "_eta_{0}".format(args.eta)
        if args.skip != 50:
            rslt_path = rslt_path + "_skip_{0}".format(args.skip)
        if args.approxdiff != "STEP":
            rslt_path = rslt_path.replace(
                "x4",
                "x4_approx_{0}_schedule_{1}".format(args.approxdiff, args.schedule),
            )
        lr_path = rslt_path.replace("results", "lr_sample")
        mkdirs([rslt_path, lr_path])

        psnrs, ssims, run_times, losses = [], [], [], []
        print("approxdiff: {0}, schedule: {1}".format(args.approxdiff, args.schedule))
        for batch in tqdm(loader, total=len(loader.dataset)):
            if args.datatype in ("LRHR_paired", "HR_downsampled"):
                lr, hr, name = batch
                batch = (lr.to(device), hr.to(device), name)
            elif args.datatype == "LR_only":
                lr, name = batch
                batch = (lr.to(device), name)
            elif args.datatype == "HR_only":
                hr, name = batch
                h, w = hr.shape[2:]
                h_, w_ = h % scale, w % scale
                hr = hr[:, :, : (h - h_), : (w - w_)]
                batch = (hr.to(device), name)
            else:
                raise "Unknown datatype"

            # do test
            with torch.no_grad():
                if args.datatype == "LR_only":
                    rslt = model.test_step_lr_only(batch, 1)
                else:
                    rslt = model.test_step(
                        batch, 1, approxdiff=args.approxdiff, schedule=args.schedule
                    )

            file_path = os.path.join(rslt_path, rslt["name"])
            lr_file_path = os.path.join(lr_path, rslt["name"])

            # print(rslt)
            # break
            if "log_img_sr" in rslt.keys():
                plt.imsave(file_path, rslt["log_img_sr"])
            if "log_img_lr" in rslt.keys():
                plt.imsave(lr_file_path, rslt["log_img_lr"])
            if "ctx" in rslt.keys():
                plt.imsave(file_path.replace(".png", "_ctx.png"), rslt["ctx"])
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

        if "HR" in args.datatype:
            hr_path = "load/benchmark/{0}/HR".format(dataset_name)
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
    parser.add_argument(
        "--datasets", default="sr-geo-15", type=str, help="dataset names"
    )
    parser.add_argument(
        "--datatype",
        default="HR_only",
        type=str,
        help="dataset type, options: (HR_only, LR_only, LRHR_paired)",
    )
    parser.add_argument("--skip", type=int, default=50)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=2.4)
    parser.add_argument("--random_seed", action="store_true")
    parser.add_argument("--lambda1", type=float, default=0.6)
    parser.add_argument("--lambda2", type=float, default=4)
    parser.add_argument("--theta", type=float, default=0)
    parser.add_argument("--noise", type=float, default=0)
    parser.add_argument("--approxdiff", default="STEP")
    parser.add_argument("--schedule", default="linear")

    return parser


test_parser = getTestParser()

if __name__ == "__main__":
    args = test_parser.parse_args()
    test(args)
