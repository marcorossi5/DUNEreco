# This file is part of DUNEdn by M. Rossi
import os
import argparse
from pathlib import Path
from shutil import copyfile
from time import time as tm
import numpy as np
import random

import torch
import torch.distributed as dist

from dunedn.denoising.dataloader import CropLoader, PlaneLoader
from dunedn.networks.models import GCNN_Net
from dunedn.denoising.args import Args
from dunedn.utils.utils import print_summary_file
from dunedn.denoising.train import train
from dunedn.utils.utils import get_configcard


def set_random_seed(random_seed=0):
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def add_arguments_distributed_training(parser):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configcard", type=Path, help="yaml config file path", default="default_config.yaml"
    )
    parser.add_argument("--local_rank", default=0, type=int, help="Distributed utility")
    parser.add_argument(
        "--local_world_size", default=1, type=int, help="Distributed utility"
    )
    parser.set_defaults(func=training_distributed)


def training_distributed(args):
    args = vars(args)
    args.pop("func")
    dist.init_process_group(backend="nccl")
    parameters = get_configcard(args.configcard)
    parameters["local_rank"] = args.local_rank
    parameters["local_world_size"] = args.local_world_size
    parameters["rank"] = dist.get_rank()
    parameters.update(args)
    args = Args(**parameters)
    args.build_directories()
    if args.rank == 0:
        print_summary_file(args)
    start = tm()
    main_distributed_training(args)
    if args.rank == 0:
        print(f"[{os.getpid()}] Process done in {tm()-start}")
        copyfile(args.runcard, args.dir_output / "input_runcard.yaml")
    dist.destroy_process_group()


def main_distributed_training(args):
    n = torch.cuda.device_count() // args.local_world_size
    args.dev_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))
    # load datasets
    set_random_seed(0)
    train_data = CropLoader(args)
    val_data = PlaneLoader(args, "val")
    # model
    model = GCNN_Net(args)
    # train
    return train(args, train_data, val_data, model)
