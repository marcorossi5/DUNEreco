"""
This module computes inference either for roi and dn, saves results and metrics
"""
import os
import sys
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib as mpl
import matplotlib.pyplot as plt
import time as tm

from args import Args
from dataloader import PlaneLoader
from model import  *
from model_utils import MyDataParallel
from model_utils import split_img
from model_utils import recombine_img
from model_utils import plot_wires

from train import test_epoch

import ssim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr
from utils.utils import get_freer_gpu
from utils.utils import moving_average

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir_name", "-p", default="../datasets/denoising",
                    type=str, help='Directory path to datasets')
PARSER.add_argument("--model", "-m", default="CNN", type=str,
                    help="either CNN or GCNN")
PARSER.add_argument("--device", "-d", default="0", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")
PARSER.add_argument("--loss_fn", "-l", default="ssim_l2", type=str,
                    help="mse, ssim, ssim_l1, ssim_l2")
PARSER.add_argument("--out_name", default=None, type=str,
                    help="Output directory")
PARSER.add_argument("--warmup", default='dn', type=str,
                    help="roi / dn")
PARSER.add_argument("--load_path", default=None, type=str,
                    help="torch .dat file to load the model")
PARSER.add_argument("--threshold", "-t", default=3, type=float,
                    help="Cut threshold on labels")


def inference(args, model, channel):
    """
    This function tests the model against
    one kind of planes and plots
    planes, histograms, and wire signals
    Parameters:
        args: Args object
        model: nn.Module object
        channel: str, either 'collection' or 'readout'

    Outputs:
        np array of metrics
    """
    #load dataset
    print('[+] Inference')
    ploader = PlaneLoader(args, 'test', 'collection')
    test_data = DataLoader(ploader,num_workers=args.num_workers)
    if args.warmup == 'roi':
        labels = ploader.clear[:,:1]
    elif args.warmup == 'dn':
        labels = ploader.clear[:,1:2]

    metrics, res, t = test_epoch(args, None, test_data, model,
                                 ana=True, warmup=args.warmup,
                                 labels=labels)

    #save results for further testing
    fname = os.path.join(args.dir_final_test, f'{args.warmup}_test_metrics')
    np.save(fname, metrics)
    print('Inference metrics saved at:', fname)

    fname = os.path.join(args.dir_final_test, f'{args.warmup}_test_res')
    np.save(fname, res)
    print('Inference results saved at:', fname)

    fname = os.path.join(args.dir_final_test, f'{args.warmup}_test_timings')
    np.save(fname, t)
    print('Inference timings saved at:', fname)

    return metrics


def main(args):
    mpl.rcParams.update({'font.size': 22})
    
    model = eval('get_' + args.model)(args)
    model = MyDataParallel(model, device_ids=args.dev_ids)
    model = model.to(args.device)
    model.eval()

    #loading model
    if args.load_path is not None:
        fname = args.load_path
    else:
        bname = os.path.join(args.dir_final_test, 'best_model.txt')
        with open(bname, 'r') as f:
            fname = f.read().strip('\n')
            f.close()
    model.load_state_dict(torch.load(fname))

    #make_plots(args)
    start = tm.time()
    metrics = inference(args, model,'collection')

    print('Final test time: %.4f\n'%(tm.time()-start))

    if args.warmup == 'roi':
            print('Final test loss: %.5f +/- %.5f'%(metrics[0], metrics[1]))
	
    else:
        print('Final test ssim: %.5f +/- %.5f'%(metrics[2], metrics[3]))
        print('Final test psnr: %.5f +/- %.5f'%(metrics[4], metrics[5]))
        print('Final test mse: %.5f +/- %.5f'%(metrics[6], metrics[7]))

    return metrics[0], metrics[1]
    
if __name__ == '__main__':
    args = vars(PARSER.parse_args())
    dev = 0

    if torch.cuda.is_available():
        if int(args['device']) == -1:
            gpu_num = get_freer_gpu()
            dev = torch.device('cuda:{}'.format(gpu_num))
        if  int(args['device']) > -1:
            dev = torch.device('cuda:{}'.format(args['device']))
        else:
            dev = torch.device('cpu')
    else:
        dev = torch.device('cpu')
    args['device'] = dev
    args['epochs'] = None
    args['loss_fn'] = "_".join(["loss", args['loss_fn']])
    print('Working on device: {}\n'.format(args['device']))
    args = Args(**args)
    assert args.warmup in ['roi', 'dn']
    start = tm.time()
    main(args)
    print('Program done in %f'%(tm.time()-start))
