"""This module contains the main denoise function"""

import os
import sys
import argparse
import time as tm

import torch

from dataloader import CropLoader
from dataloader import CropValLoader
from dataloader import PlaneLoader
#from dataloader import load_planes
from model import  *
from args import Args

from model_utils import MyDataParallel
from model_utils import print_summary_file

import train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_freer_gpu

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir_name", "-p", default="../datasets",
                    type=str, help='Directory path to datasets')
PARSER.add_argument("--epochs", "-n", default=50, type=int,
                    help="training epochs")
PARSER.add_argument("--model", "-m", default="CNN", type=str,
                    help="either CNN or GCNN")
PARSER.add_argument("--device", "-d", default="0", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")


def main(args):
    """This is the main function"""
    torch.cuda.set_enabled_lms(True)
    print_summary_file(args)
    #load datasets
    train_data = torch.utils.data.DataLoader(CropLoader(args, "train"),
                                            shuffle=True,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    test_data = torch.utils.data.DataLoader(CropLoader(args, "val"),
                                            shuffle=True,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    
    #build model
    model = eval('get_' + args.model)(args.k,
                                      args.in_channels,
                                      args.hidden_channels,
                                      args.crop_size)
    model = MyDataParallel(model, device_ids=args.dev_ids)
    model = model.to(args.device)

    #train
    train.train(args, train_data, test_data, model)

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
    print('Working on device: {}\n'.format(ARGS['device']))
    ARGS = Args(**ARGS)
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
