"""This module contains the main denoise function"""

import os
import sys
import argparse
import shutil
import time as tm

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import DataLoader

from distributed import set_random_seed

from dataloader import CropLoader
from dataloader import PlaneLoader
from model import DenoisingModel
from args import Args

from model_utils import print_summary_file
from model_utils import weight_scan

import train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_freer_gpu
from utils.utils import load_yaml

def main(args):
    """This is the main function"""

    n = torch.cuda.device_count() // args.local_world_size
    args.dev_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))

    #load datasets
    set_random_seed(0)
    train_data = CropLoader(args,'train','collection')
    val_data = PlaneLoader(args,'val','collection')

    model = DenoisingModel(args)

    #train
    return train.train(args, train_data, val_data, model)


def spmd_main(card, local_rank, local_world_size):
    """ Spawn distributed processes """
    dist.init_process_group(backend="nccl")

    # load configuration
    parameters = load_yaml(card)
    parameters["local_rank"] = local_rank
    parameters["local_world_size"] = local_world_size
    parameters["rank"] = dist.get_rank()
    args = Args(**parameters)
    args.build_directories(build=( args.rank==0 ))
    if args.rank == 0:
        print_summary_file(args)
    main(args)
    if args.rank==0:
        src = "/nfs/public/romarco/DUNEreco/denoising/logdir"
        shutil.copytree(src, os.path.join(args.dir_output, "logdir"))
    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--card", type=str, help='yaml config file path',
                        default="/nfs/public/romarco/DUNEreco/denoising/configcards/default_config.yaml")
    parser.add_argument("--local_rank", default=0, type=int,
                    help="Distributed utility")
    parser.add_argument("--local_world_size", default=1, type=int,
                    help="Distributed utility")
    # load configuration
    args = vars(parser.parse_args())
    # main
    START = tm.time()
    spmd_main(**args)
    print(f'[{os.getpid()}] Process done in {tm.time()-START}')
        
