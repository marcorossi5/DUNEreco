"""This module contains the main denoise function"""

import os
import sys
import argparse
import time as tm

import torch

from dataloader import CropLoader
from dataloader import PlaneLoader
#from dataloader import load_planes
from model import  *
from args import Args

from model_utils import MyDataParallel
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
PARSER.add_argument("--model", "-m", default="CNN", type=str,
                    help="CNN, CNNv2, GCNN, GCNNv2")
PARSER.add_argument("--device", "-d", default="0", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")
PARSER.add_argument("--loss_fn", "-l", default="ssim", type=str,
                    help="mse, ssim, ssim_l1, ssim_l2")
PARSER.add_argument("--lr", default=0.009032117010326078, type=float,
                    help="training epochs")
PARSER.add_argument("--out_name", default=None, type=str,
                    help="Output directory")

def freeze_weights(model, ROI):
    """
    Freezes weights of ROI either finder or GCNN denoiser
    Parameters:
        model: torch.nn.Module, first childred should be ROI
        ROI: either 1 (freezes ROI) or 0 (freezes denoiser)
    """
    for i, child in enumerate(model.children()):
        if ((i == 0)%2 + ROI + 1)%2:
            for param in child.parameters():
                param.requires_grad = False
    return model

def warmup_trains(args, train_data, test_data, mode):
    """
    Wrapper for warmup trains
    Parameters:
        mode: either 1 (freezes ROI) or 0 (freezes denoiser)
        fname: saved model to load
    """
    model = eval('get_' + args.model)(args)
    model = freeze_weights(model, mode)
    model = MyDataParallel(model, device_ids=args.dev_ids)
    model = model.to(args.device)

    if mode == 0 :
        warmup = 'roi'
    if mode == 1:
        warmup = 'dn'

    train.train(args, train_data, test_data, model, warmup=warmup)

def main(args):
    """This is the main function"""
    torch.cuda.set_enabled_lms(True)
    print_summary_file(args)
    #load datasets
    train_data = torch.utils.data.DataLoader(CropLoader(args,'train','collection'),
                                            shuffle=True,
                                            batch_size=args.batch_size,
                                            num_workers=args.num_workers)
    test_data = torch.utils.data.DataLoader(PlaneLoader(args,'val','collection'),
                                            num_workers=args.num_workers)
    #build model
    epochs = args.epochs
    epoch_test = args.epoch_test
    args.epochs = args.warmup_roi_epochs
    args.epoch_test = 999999999

    warmup_trains(args, train_data, test_data, 0)
    
    args.epochs = args.warmup_dn_epochs
    args.load = True
    args.load_epoch = args.warmup_roi_epochs

    warmup_trains(args, train_data, test_data, 1)

    args.epochs = epochs
    args.epoch_test = epoch_test
    args.load_epoch = args.warmup_dn_epochs

    model = eval('get_' + args.model)(args)
    model = MyDataParallel(model, device_ids=args.dev_ids)
    model = model.to(args.device)

    #train
    return train.train(args, train_data, test_data, model, warmup=False)    

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
