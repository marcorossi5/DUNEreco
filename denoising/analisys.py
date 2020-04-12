import os
import sys
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
import time as tm

from args import Args
from dataloader import PlaneLoader
from model import  *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr

parser = argparse.ArgumentParser()
parser.add_argument("--dir_name", "-p", default="../datasets",
                    type=str, help='Directory path to datasets')
parser.add_argument("--epochs", "-n", default=0, type=int,
                    help="training epochs")
parser.add_argument("--model", "-m", default="CNN", type=str,
                    help="either CNN or GCNN")
parser.add_argument("--device", "-d", default="-1", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")

def final_test(args, test_data, model):
    model.eval()
    mse_loss = torch.nn.MSELoss()
    psnr = []
    mse = []
    print('Number of planes to be tested:', len(test_data))
    for (clear, noised) in test_data:
        res = model.forward_image(noised, args.device, args.test_batch_size)
        psnr.append(compute_psnr(clear, res))
        mse.append(mse_loss(clear, res).item())
    
    #printing a single plane
    fname = os.path.join(args.dir_final_test, 'final_test.png')
    fig = plt.figure(figsize=(20,25))
    plt.suptitle('Final test denoising example')
    ax = fig.add_subplot(311)
    ax.title.set_text('Noised image')
    z = ax.imshow(noised[0,0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(312)
    ax.title.set_text('Clear image')
    z = ax.imshow(clear[0,0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(313)
    ax.title.set_text('Denoised image')
    z = ax.imshow(res[0,0])
    fig.colorbar(z, ax=ax)
    plt.savefig(fname)
    plt.close()

    return np.array([np.mean(psnr), np.std(psnr)/np.sqrt(len(psnr)),
                np.mean(mse), np.std(mse)/np.sqrt(len(psnr))])

def make_plots(args):
    fname = os.path.join(args.dir_metrics, 'loss_sum.npy')
    loss_sum = np.load(fname)

    fname = os.path.join(args.dir_metrics, 'test_epochs.npy')
    test_epochs = np.load(fname)

    fname = os.path.join(args.dir_metrics, 'test_metrics.npy')
    test_metrics = np.load(fname)

    fname = os.path.join(args.dir_metrics, 'metrics.png')
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(121)
    ax.title.set_text('Metrics')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics')
    ax.plot(loss_sum, label='train_loss')
    ax.errorbar(test_epochs,test_metrics[2],
                yerr=test_metrics[3],label='test_loss')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.title.set_text('PSNR')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PSNR [dB]')
    ax.errorbar(test_epochs,test_metrics[0],
                yerr=test_metrics[1])
    plt.savefig(fname)
    plt.close()
    print('saved image at: %s'%fname)

def main(args):
    mpl.rcParams.update({'font.size': 22})
    #load datasets
    test_data = torch.utils.data.DataLoader(PlaneLoader(args.dataset_dir,
                                                      'collection_test'
                                                      ),
                                        shuffle=True,
                                        batch_size=1,
                                        num_workers=args.num_workers)
    
    model = eval('get_' + args.model)(args.k,
                                args.in_channels,
                                args.hidden_channels
                                ).to(args.device)
    start = tm.time()
    res = final_test(args, test_data, model)
    print('Final test time: %.4f\n'%(tm.time()-start))
    print('Final test psnr: %.4f +/- %.4f'%(res[0], res[1]))
    print('Final test loss: %.4f +/- %.4f'%(res[2], res[3]))

    fname = os.path.join(args.dir_final_test, 'best_model.txt')
    with open(fname, 'r') as f:
        lname = f.read()
        f.close()
    model.load_state_dict(torch.load(lname))

    make_plots(args)

if __name__ == '__main__':
    args = vars(parser.parse_args())
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
    print('Working on device: {}\n'.format(args['device']))
    args = Args(**args)
    start = tm.time()
    main(args)
    print('Program done in %f'%(tm.time()-start))