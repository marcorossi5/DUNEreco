"""This module contains the main denoise function"""

import os
import sys
import argparse
import time as tm

import torch
import numpy as np

from dataloader import CropLoader
from dataloader import PlaneLoader
#from dataloader import load_planes
from model import  *
from args import Args

from model_utils import print_summary_file
from model_utils import weight_scan

import train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_freer_gpu

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir_name", "-p", default="../datasets/denoising",
                    type=str, help='Directory path to datasets')
PARSER.add_argument("--epochs", "-n", default=50, type=int,
                    help="training epochs")
PARSER.add_argument("--model", "-m", default="GCCNNv2", type=str,
                    help="CNN, CNNv2, GCNN, GCNNv2")
PARSER.add_argument("--device", "-d", default="0", type=str,
                    help="-1 (automatic) / -2 (cpu) / gpu number")
PARSER.add_argument("--loss_fn", "-l", default="ssim_l2", type=str,
                    help="mse, ssim, ssim_l1, ssim_l2")
PARSER.add_argument("--lr", default=0.009032117010326078, type=float,
                    help="training epochs")
PARSER.add_argument("--out_name", default=None, type=str,
                    help="Output directory")
PARSER.add_argument("--load_path", default=None, type=str,
                    help="torch .dat file to load the model")
PARSER.add_argument("--warmup", default='dn', type=str,
                    help="roi / dn")


def main(args):
    """This is the main function"""
    torch.cuda.set_enabled_lms(True)
    print_summary_file(args)
    #load datasets
    train_data = torch.utils.data.DataLoader(CropLoader(args,'train','collection'),
                                            shuffle=True,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    test_data = PlaneLoader(args,'val','collection')
    
    
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

if __name__ == '__main__':
    ARGS = vars(PARSER.parse_args())
    DEV = 0

    if torch.cuda.is_available():
        if int(ARGS['device']) == -1:
            GPU_NUM = get_freer_gpu()
            DEV = torch.device('cuda:{}'.format(GPU_NUM))
        elif  int(ARGS['device']) > -1:
            DEV = torch.device('cuda:{}'.format(ARGS['device']))
        else:
            DEV = torch.device('cpu')
    else:
        DEV = torch.device('cpu')
    ARGS['device'] = DEV
    ARGS['loss_fn'] = "_".join(["loss", ARGS['loss_fn']])
    print('Working on device: {}\n'.format(ARGS['device']))
    ARGS = Args(**ARGS)
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
