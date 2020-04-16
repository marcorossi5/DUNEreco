import os
import sys
import argparse

import numpy as np
import torch

import time as tm

from dataloader import CropLoader
from dataloader import PlaneLoader
#from dataloader import load_planes
from model import  *
from args import Args

import train

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import get_freer_gpu

parser = argparse.ArgumentParser()
parser.add_argument("--dir_name", "-p", default="../datasets",
                    type=str, help='Directory path to datasets')
parser.add_argument("--epochs", "-n", default=10, type=int,
                    help="training epochs")
parser.add_argument("--model", "-m", default="CNN", type=str,
                    help="either CNN or GCNN")
parser.add_argument("--device", "-d", default="-1", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")


def main(args):
	torch.cuda.set_enabled_lms(True)
    #load datasets
    train_data = torch.utils.data.DataLoader(CropLoader(args.dataset_dir),
                                        shuffle=True,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers)
    val_data = [torch.utils.data.DataLoader(PlaneLoader(args.dataset_dir,
                                                      'collection_val'
                                                      ),
                                        num_workers=args.num_workers),
    			torch.utils.data.DataLoader(PlaneLoader(args.dataset_dir,
                                                      'readout_val'
                                                      ),
                                        num_workers=args.num_workers)]
    
    #build model
    model = eval('get_' + args.model)(args.k,
                                args.in_channels,
                                args.hidden_channels
                                ).to(args.device)
    #train
    train.train(args, train_data, val_data, model)

if __name__ == '__main__':
    args = vars(parser.parse_args())
    dev = 0

    if torch.cuda.is_available():
        if int(args['device']) == -1:
            gpu_num = get_freer_gpu()
            dev = torch.device('cuda:{}'.format(gpu_num))
        elif  int(args['device']) > -1:
            dev = torch.device('cuda:{}'.format(args['device']))
        else:
            dev = torch.device('cpu')
    else:
        dev = torch.device('cpu')
    args['device'] = dev
    print('Working on device: {}\n'.format(args['device']))
    args = Args(**args)
    start = tm.time()
    main(args)
    print('Program done in %f'%(tm.time()-start))