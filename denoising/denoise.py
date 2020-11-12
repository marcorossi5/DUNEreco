"""This module contains the main denoise function"""

import os
import sys
import argparse
import time as tm

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler 
from torch.nn.parallel import DistributedDataParallel
import numpy as np

from distributed import set_random_seed

from dataloader import CropLoader
from dataloader import PlaneLoader
from model import  *
from args import Args

from model_utils import print_summary_file
from model_utils import weight_scan

import train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_freer_gpu


def main(args):
    """This is the main function"""

    n = torch.cuda.device_count() // args.local_world_size
    args.dev_ids = list(range(args.local_rank * n, (args.local_rank + 1) * n))

    #print(
    #    f"[{os.getpid()}] rank = {dist.get_rank()}, "
    #    + f"world_size = {dist.get_world_size()}, n = {n}, device_ids = {device_ids}"
    #)

    #load datasets
    set_random_seed(0)
    train_set = CropLoader(args,'train','collection')
    train_sampler = DistributedSampler(dataset=train_set)
    train_loader = DataLoader(dataset=train_set, sampler=train_sampler,
                              shuffle=True, batch_size=args.batch_size,
                              num_workers=args.num_workers)
    
    test_set = PlaneLoader(args,'val','collection')
    if args.warmup == 'roi':
        labels = test_data.clear[:,1:2]
    if args.warmup == 'dn':
        labels = test_data.clear[:,:1]
    test_data = torch.utils.data.DataLoader(test_data,
                                            num_workers=args.num_workers)

    model = eval('get_' + args.model)(args)

    #train
    return train.train(args, train_data, test_data,
                model, warmup=args.warmup, labels=labels)


def spmd_main(args):
    #env_dict = {
    #    key: os.environ[key]
    #    for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE")
    #}
    #print(f"[{os.getpid()}] Initializing process group with: {env_dict}")
    dist.init_process_group(backend="nccl")
    #print(
    #    f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
    #    + f"rank = {args.rank}, backend={dist.get_backend()}"
    #)

    main(args)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_name", "-p", default="../datasets/denoising",
                    type=str, help='Directory path to datasets')
    parser.add_argument("--epochs", "-n", default=50, type=int,
                    help="training epochs")
    parser.add_argument("--model", "-m", default="GCNN", type=str,
                    help="CNN, GCNN")
    parser.add_argument("--loss_fn", "-l", default="ssim_l2", type=str,
                    help="mse, ssim, ssim_l1, ssim_l2")
    parser.add_argument("--lr", default=0.009032117010326078, type=float,
                    help="training epochs")
    parser.add_argument("--out_name", default=None, type=str,
                    help="Output directory")
    parser.add_argument("--load_path", default=None, type=str,
                    help="torch .dat file to load the model")
    parser.add_argument("--warmup", default='dn', type=str,
                    help="roi / dn")
    parser.add_argument("--local_rank", default=0, type=int,
                    help="Distributed utility")
    parser.add_argument("--local_world_size", default=1, type=int,
                    help="Distributed utility")

    args = parser.parse_args()
    args.rank = dist.get_rank()
    args.loss_fn = "_".join(["loss", args.loss_fn])
    args = Args(**args)
    if args.rank == 0:
        print_summary_file(args)
    START = tm.time()
    spmd_main(args)
    print(f'[{os.getpid()}] Program done in {tm.time()-START}')
