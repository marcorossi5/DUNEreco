import argparse
import time as tm
import numpy as np
import torch
import matplotlib.pyplot as plt

from args import Args

from ssim import stat_ssim

from dataloader import PlaneLoader

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--dir_name",
    "-p",
    default="../datasets/denoising",
    type=str,
    help="Directory path to datasets",
)
PARSER.add_argument(
    "--device",
    "-d",
    default="0",
    type=str,
    help="-1 (automatic)/ -2 (cpu) / gpu number",
)


def main(args):
    """Main function: plots SSIM of a batch of crops to select k1, k2 parameters"""
    data = PlaneLoader(args, "val", "collection")
    clear = data.clear.to(args.device)
    noisy = data.noisy * (data.norm[1] - data.norm[0]) + data.norm[0]
    noisy = noisy.to(args.device)

    print("Number of planes: ", len(clear))
    print("MSE: ", torch.nn.MSELoss()(noisy, clear))

    y = []
    x = np.logspace(-15, -1, 10000)

    for i in x:
        y.append(
            stat_ssim(noisy, clear, data_range=1.0, size_average=True, K=(i, i))
            .cpu()
            .item()
        )

    plt.figure(figsize=(15, 15))
    plt.plot(x, y)
    plt.xscale("log")
    plt.savefig("../collection_t.png")


if __name__ == "__main__":
    ARGS = vars(PARSER.parse_args())
    DEV = 0

    if torch.cuda.is_available():
        if int(ARGS["device"]) == -1:
            from utils.utils import get_freer_gpu
            GPU_NUM = get_freer_gpu()
            DEV = torch.device("cuda:{}".format(GPU_NUM))
        elif int(ARGS["device"]) > -1:
            DEV = torch.device("cuda:{}".format(ARGS["device"]))
        else:
            DEV = torch.device("cpu")
    else:
        DEV = torch.device("cpu")
    ARGS["device"] = DEV
    print("Working on device: {}\n".format(ARGS["device"]))
    ARGS["epochs"] = None
    ARGS["model"] = None
    ARGS["loss_fn"] = None
    ARGS = Args(**ARGS)
    START = tm.time()
    main(ARGS)
    print("Program done in %f" % (tm.time() - START))
