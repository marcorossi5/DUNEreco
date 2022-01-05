"""
This module computes recob::wires results for planes in the test set.

Threshold is the cut below which a clear raw digit is considered an
hit with signal.

Output file contains ssim, psnr and mse provided with mean values
and uncertainties.
"""
import os
import argparse
from pathlib import Path
import numpy as np
from time import time as tm
import torch
from dunedn.utils.utils import compute_psnr
from dunedn.denoising.losses import loss_mse, loss_ssim


def main(dirname, device):
    # Load targets: raw::RawDigits w/o noise
    file_name = dirname / "planes/collection_clear.npy"
    digits = np.load(file_name)

    # Load recob::wires
    file_name = dirname / "benchmark/wires/pandora_collection_wires.npy"
    wires = np.load(file_name)

    # input images should be 4-d tensors
    digits = torch.Tensor(digits[:, None]).to(device)
    wires = torch.Tensor(wires[:, None]).to(device)

    ssim = []
    mse = []
    psnr = []

    for digit, wire in zip(digits, wires):
        ssim.append(1 - loss_ssim()(digit, wire).cpu().item())
        mse.append(loss_mse()(digit, wire).cpu().item())
        psnr.append(compute_psnr(digit, wire))

    ssim_mean = np.mean(ssim)
    ssim_std = np.std(ssim) / np.sqrt(len(ssim))

    mse_mean = np.mean(mse)
    mse_std = np.std(mse) / np.sqrt(len(mse))

    psnr_mean = np.mean(psnr)
    psnr_std = np.std(psnr) / np.sqrt(len(psnr))

    res = np.array([[ssim_mean, ssim_std], [psnr_mean, psnr_std], [mse_mean, mse_std]])
    fname = f"denoising/benchmarks/results/pandora_wires_metrics"
    np.save(fname, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirname",
        default="../datasets/backup/test",
        type=Path,
        help="Directory path to datasets",
    )
    parser.add_argument("--device", default="0", help="gpu number")
    args = vars(parser.parse_args())
    gpu = torch.cuda.is_available()
    dev = args["device"]
    dev = f"cuda:{dev}" if gpu else "cpu"
    args["device"] = torch.device(dev)
    START = tm.time()
    main(**args)
    print("Program done in %f" % (tm.time() - START))
