import json
from os import path as osp

import numpy as np
import pandas as pd
import pyiqa
import torch
from litsr.metrics import calc_fid, calc_psnr_ssim
from litsr.transforms import uint2single
from litsr.utils import load_file_list, read_images

root = "results"
# rslt_file_name = "rslt_major_revis.xlsx"
rslt_file_name = "rslt_fastdpm.xlsx"
# datasets = ["sr-geo-15", "Google-15"]
datasets = ["sr-geo-15"]

# methodList = [
#     "Bicubic",
#     "SAN",
#     "ESRGAN",
#     "ZSSR",
#     "DASR",
#     "RealESRGAN",
#     "BSRGAN",
#     "IKC",
#     "OURS-v1",
# "OURS-wo-c",
# ]
methodList = [
    "OURS_FASTDPM_VAR_quadratic",
]
degradationList = [
    "iso_sig_0.0_noise_0",
    "iso_sig_1.2_noise_0",
    "iso_sig_2.4_noise_0",
    "iso_sig_3.6_noise_0",
    "iso_sig_4.8_noise_0",
    "iso_sig_2.4_noise_0_skip_1",
    "iso_sig_2.4_noise_0_skip_10",
    "iso_sig_2.4_noise_0_skip_25",
    "iso_sig_2.4_noise_0_skip_100",
    "iso_sig_2.4_noise_0_skip_200",
    "iso_sig_2.4_noise_0_skip_500",
    #     # "l1_1.2_l2_2.4_theta_0.0_noise_0.0",
    #     # "l1_1.2_l2_2.4_theta_0.0_noise_5.0",
    #     # "l1_1.2_l2_2.4_theta_0.0_noise_10.0",
    #     # "l1_1.2_l2_2.4_theta_45.0_noise_0.0",
    #     # "l1_1.2_l2_2.4_theta_45.0_noise_5.0",
    #     # "l1_1.2_l2_2.4_theta_45.0_noise_10.0",
    #     # "l1_2.4_l2_1.2_theta_45.0_noise_0.0",
    #     # "l1_2.4_l2_1.2_theta_45.0_noise_5.0",
    #     # "l1_2.4_l2_1.2_theta_45.0_noise_10.0",
    #     # "l1_3.6_l2_2.4_theta_0.0_noise_0.0",
    #     # "l1_3.6_l2_2.4_theta_0.0_noise_5.0",
    #     # "l1_3.6_l2_2.4_theta_0.0_noise_10.0",
]
# degradationList = [""]

force_recalc = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
iqa_lpips = pyiqa.create_metric("lpips").to(device)
iqa_niqe = pyiqa.create_metric("niqe").to(device)
iqa_brisque = pyiqa.create_metric("brisque").to(device)


def writeRslt(rsltPath, rslt):
    with open(osp.join(rsltPath, "rslt.txt"), "w") as f:
        f.write(json.dumps(rslt))


def readRslt(rsltPath):
    with open(osp.join(rsltPath, "rslt.txt"), "r") as f:
        rslt = json.load(f)
    return rslt


def handleDataset(dataset):
    hrPath = osp.join(root, dataset, "HR")
    hrs = read_images(load_file_list(hrPath, ".*.png"))
    rsltList = []
    for degradName in degradationList:
        for methodName in methodList:
            print("--" * 10)
            print("{0} | {1} | {2}".format(dataset, degradName, methodName))
            rsltPath = osp.join(root, dataset, methodName, degradName)
            if osp.exists(osp.join(rsltPath, "rslt.txt")) and (not force_recalc):
                rslt = readRslt(rsltPath)
                rslt["methodName"] = methodName
            else:
                rsltPathList = load_file_list(rsltPath, ".*.png")
                rslts = read_images(rsltPathList)
                psnrList = []
                ssimList = []

                for hr, sr in zip(hrs, rslts):
                    psnr, ssim = calc_psnr_ssim(hr, sr, crop_border=4, test_Y=False)
                    psnrList.append(psnr)
                    ssimList.append(ssim)

                fid = calc_fid([hrPath, rsltPath])
                hr_tensor = torch.from_numpy(
                    uint2single(np.array(hrs).transpose(0, 3, 1, 2))
                ).to(device)
                rslt_tensor = torch.from_numpy(
                    uint2single(np.array(rslts).transpose(0, 3, 1, 2))
                ).to(device)

                rslt = {
                    "methodName": methodName,
                    "degradName": degradName,
                    "psnr": np.array(psnrList).mean(),
                    "ssim": np.array(ssimList).mean(),
                    "fid": fid,
                    "lpips": iqa_lpips(rslt_tensor, hr_tensor).mean().item(),
                    "niqe": iqa_niqe(rslt_tensor).mean().item(),
                    "brisque": iqa_brisque(rslt_tensor).mean().item(),
                }
            rsltList.append(rslt)
            writeRslt(rsltPath, rslt)
            print(rslt)
    df = pd.DataFrame(rsltList)
    df.to_excel(osp.join(root, dataset, rslt_file_name), index=False)


for dataset in datasets:
    handleDataset(dataset)
