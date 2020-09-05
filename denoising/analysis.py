"""This module computes inference either for roi and dn, saves results and metrics"""
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
    test_data = DataLoader(PlaneLoader(args, 'test', 'collection'),
                                       num_workers=args.num_workers)

    metrics, res, t = test_epoch(args, None, test_data, model,
                                 ana=True, warmup=args.warmup)

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

    '''
    diff = np.abs(res-clear)

    fname = os.path.join(args.dir_final_test, f'{channel}_residuals.png')
    fig = plt.figure(figsize=(20,25))
    plt.suptitle(f'Final denoising test on {channel} planes')

    ax = fig.add_subplot(411)
    ax.title.set_text(r'Sample of $I_{Clear}$ image')
    z = ax.imshow(clear[0,0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(412)
    ax.title.set_text(r'Sample of $I_{DN}$ image')
    z = ax.imshow(res[0,0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(413)
    ax.title.set_text(r'Sample of $|I_{DN} - I_{Clear}|$')
    z = ax.imshow(diff[0,0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(427)
    ax.hist(diff[0].flatten(), 100, density=True)
    ax.set_yscale('log')
    ax.title.set_text(r'Sample of histogram of $|I_{DN} - I_{Clear}|$')

    ax = fig.add_subplot(428)
    ax.hist(diff.flatten(), 100, density=True)
    ax.set_yscale('log')
    ax.title.set_text(r'Histogram of all $|I_{DN} - I_{Clear}|$')

    plt.savefig(fname, bbox='tight')
    plt.close()
    print(f'Saved residuals at {fname}')

    sample = torch.randint(0, clear.shape[0],(25,))
    wire = torch.randint(0, clear.shape[-2],(25,))

    plot_wires(args.dir_final_test,
               clear,
               f"{channel}_label",
               sample,
               wire)
    plot_wires(args.dir_final_test,
               res,
               f"{channel}_DN",
               sample,
               wire)
    plot_wires(args.dir_final_test,
               noisy,
               f"{channel}_noisy",
               sample,
               wire)

    return x
    '''


'''
def make_plots(args):
    fname = os.path.join(args.dir_metrics, 'loss_sum.npy')
    loss_sum = np.load(fname)

    #smoothing the loss
    weight = 0
    #weight = 2/(len(loss_sum)+1)
    loss_avg = moving_average(loss_sum[0], weight)
    #perc_avg = moving_average(loss_sum[1], weight)

    fname = os.path.join(args.dir_metrics, 'test_epochs.npy')
    test_epochs = np.load(fname)

    fname = os.path.join(args.dir_metrics, 'test_metrics.npy')
    test_metrics = np.load(fname)


    fname = os.path.join(args.dir_metrics, 'metrics.png')
    fig = plt.figure(figsize=(20,30))
    ax = fig.add_subplot(121)
    ax.title.set_text('Metrics')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics')
    ax.plot(loss_avg, color='g', label='train loss')
    ax.plot(loss_sum[0], color='g', alpha=0.2)
    #ax.plot(perc_avg, color='r', label='perc loss')
    #ax.plot(loss_sum[1], color='r', alpha=0.2)
    ax.errorbar(test_epochs,test_metrics[0],
                yerr=test_metrics[1], label='val loss')
    #ax.errorbar(test_epochs,test_metrics[4],
    #            yerr=test_metrics[5], label='test mse')
    ax.set_yscale('log')
    ax.legend()

    ax = fig.add_subplot(322)
    ax.title.set_text('validation ssim')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('ssim')
    ax.errorbar(test_epochs,test_metrics[2],
                yerr=test_metrics[3])

    ax = fig.add_subplot(324)
    ax.title.set_text('validation pSNR')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('pSNR [dB]')
    ax.errorbar(test_epochs,test_metrics[4],
                yerr=test_metrics[5])

    ax = fig.add_subplot(326)
    ax.title.set_text('validation mse')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('mse')
    ax.errorbar(test_epochs,test_metrics[6],
                yerr=test_metrics[7])
    plt.savefig(fname)
    plt.close()
    print('Saved image at: %s'%fname)

    fname = os.path.join(args.dir_metrics, 'parameters.npy')
    params = np.load(fname)

    fname = os.path.join(args.dir_metrics, 'parameters.png')
    fig = plt.figure(figsize=(40,20))
    ax = fig.add_subplot(2,4,1)
    ax.title.set_text('a0')
    ax.plot(params[0])

    ax = fig.add_subplot(2,4,2)
    ax.title.set_text('a1')
    ax.plot(params[1])

    ax = fig.add_subplot(2,4,3)
    ax.title.set_text('a2')
    ax.plot(params[2])

    ax = fig.add_subplot(2,4,4)
    ax.title.set_text('a3')
    ax.plot(params[3])

    ax = fig.add_subplot(2,4,5)
    ax.title.set_text('b0')
    ax.plot(params[4])

    ax = fig.add_subplot(2,4,6)
    ax.title.set_text('b1')
    ax.plot(params[5])

    ax = fig.add_subplot(2,4,7)
    ax.title.set_text('b2')
    ax.plot(params[6])

    ax = fig.add_subplot(2,4,8)
    ax.title.set_text('b3')
    ax.plot(params[7])

    plt.savefig(fname, bbox='tight')
    plt.close()
    print(f'Saved parameters at {fname}')
'''

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
            bname = f.read().strip('\n')
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
