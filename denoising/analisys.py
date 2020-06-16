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
from model_utils import MyDataParallel
from model_utils import split_img
from model_utils import recombine_img
from model_utils import plot_wires
import ssim



sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import compute_psnr
from utils.utils import get_freer_gpu
from utils.utils import moving_average

parser = argparse.ArgumentParser()
parser.add_argument("--dir_name", "-p", default="../datasets",
                    type=str, help='Directory path to datasets')
parser.add_argument("--epochs", "-n", default=0, type=int,
                    help="training epochs")
parser.add_argument("--model", "-m", default="CNN", type=str,
                    help="either CNN or GCNN")
parser.add_argument("--device", "-d", default="0", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")
parser.add_argument("--loss_fn", "-l", default="ssim", type=str,
                    help="mse, ssim, ssim_l1, ssim_l2")

def inference(args, model):
    #load dataset
    test_data = [torch.utils.data.DataLoader(PlaneLoader(args,
                                                      'collection_test'
                                                      ),
                                        num_workers=args.num_workers),
                 torch.utils.data.DataLoader(PlaneLoader(args,
                                                      'readout_test'
                                                      ),
                                        num_workers=args.num_workers)]

    #test_data = [load_planes(args.dataset_dir, 'collection_test'),
    #             load_planes(args.dataset_dir, 'readout_test')]
    legend = ['collection', 'readout']

    mse_loss = torch.nn.MSELoss()
    ssim_loss = []
    psnr = []
    mse = []
    res = [[],[]]
    noisy = [[],[]]
    labels = [[],[]]
    p_x, p_y = model.patch_size
    split_size = args.test_batch_size
    a = args.a
    #print('Number of planes to be tested:', len(test_data))
    for i, data in enumerate(test_data):
        for (clear, noised) in data:
            labels[i] += [clear]
            noisy[i] += [noised]
            
            crops, crops_shape, pad = split_img(noised,(p_x,p_y))
            loader = torch.split(crops, split_size)
            dn = []
            for chunk in loader:
                answer = model(chunk.to(args.device)).cpu().data
                dn.append(answer)

            dn = torch.cat(dn)

            dn = recombine_img(dn, crops_shape, pad)

            res[i] += [dn]

            #res[i] += [model.forward_image(noised,
            #                               args.device,
            #                               args.test_batch_size)]
            psnr.append(compute_psnr(clear, res[i][-1]))
            mse.append(mse_loss(clear, res[i][-1]).item())
            ssim_loss.append(model.loss_fn(clear,res[i][-1]).cpu().item())
        labels[i] = np.concatenate(labels[i])[:,0]
        noisy[i] = np.concatenate(noisy[i])[:,0] 
        res[i] = np.concatenate(res[i])
    #res[i] is a np array with shape [batch,row,col]
    #the same for labels[i]
    
    #printing a single plane
    '''
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
    '''
    diff = [np.abs(res[i] - labels[i]) for i in range(len(res))]
    #print('first', diff[0].min(), diff[0].max())
    #print('all', diff[].min(), diff.max())

    fname = os.path.join(args.dir_final_test, 'residuals.png')
    fig = plt.figure(figsize=(20,25))
    plt.suptitle('Final denoising test')

    ax = fig.add_subplot(411)
    ax.title.set_text('Sample of Clear image')
    z = ax.imshow(labels[0][0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(412)
    ax.title.set_text('Sample of Denoised image')
    z = ax.imshow(res[0][0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(413)
    ax.title.set_text('Sample of |Denoised - Clear|')
    z = ax.imshow(diff[0][0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(427)
    ax.hist([i[0].flatten() for i in diff], 100,
             stacked=True, label=legend,
             density=True, histtype='step')
    ax.set_yscale('log')
    ax.legend()
    ax.title.set_text('Sample of histogram of |Diff|')

    ax = fig.add_subplot(428)
    ax.hist([i.flatten() for i in diff], 100,
             stacked=True, label=legend,
             density=True, histtype='step')
    ax.set_yscale('log')
    ax.legend()
    ax.title.set_text('Histogram of all |Diff|')

    plt.savefig(fname)
    plt.close()

    sample = torch.randint(0, len(labels[0]),(25,))
    wire = torch.randint(0, labels[0][0].shape[0],(25,))

    plot_wires(args.dir_final_test,
               labels[0],
               "collection_label",
               sample,
               wire)
    plot_wires(args.dir_final_test,
               res[0],
               "collection_DN",
               sample,
               wire)
    plot_wires(args.dir_final_test,
               noisy[0],
               "collection_noisy",
               sample,
               wire)

    sample = torch.randint(0, len(labels[1]),(25,))
    wire = torch.randint(0, labels[1][0].shape[0],(25,))

    plot_wires(args.dir_final_test,
               labels[1],
               "readout_label",
               sample,
               wire)
    plot_wires(args.dir_final_test,
               res[1],
               "readout_DN",
               sample,
               wire)
    plot_wires(args.dir_final_test,
               noisy[1],
               "readout_noisy",
               sample,
               wire)
    '''
    fname = os.path.join(args.dir_final_test, 'wire.png')
    fig = plt.figure(figsize=(20,25))
    ax = fig.add_subplot(231)
    ax.title.set_text('Collection Wire %d DN'%800)
    ax.plot(res[0][0][800], linewidth=0.3)

    ax = fig.add_subplot(232)
    ax.title.set_text('Collection Wire %d noisy'%800)
    ax.plot(noisy[0][0][800], linewidth=0.3)

    ax = fig.add_subplot(233)
    ax.title.set_text('Collection Wire %d labels'%800)
    ax.plot(labels[0][0][800], linewidth=0.3)

    ax = fig.add_subplot(234)
    ax.title.set_text('Readout Wire %d DN'%700)
    ax.plot(res[1][0][700], linewidth=0.3)

    ax = fig.add_subplot(235)
    ax.title.set_text('Readout Wire %d noisy'%700)
    ax.plot(noisy[1][0][700], linewidth=0.3)

    ax = fig.add_subplot(236)
    ax.title.set_text('Readout Wire %d labels'%700)
    ax.plot(labels[1][0][700], linewidth=0.3)

    plt.savefig(fname)
    plt.close()
    '''

    return np.array([np.mean(ssim_loss),
                     np.std(ssim_loss)/np.sqrt(len(ssim_loss)),
                     np.mean(psnr),
                     np.std(psnr)/np.sqrt(len(psnr)),
                     np.mean(mse),
                     np.std(mse)/np.sqrt(len(psnr))])

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
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(121)
    ax.title.set_text('Metrics')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Metrics')
    ax.plot(loss_avg, color='g', label='train ssim')
    ax.plot(loss_sum[0], color='g', alpha=0.2)
    #ax.plot(perc_avg, color='r', label='perc loss')
    #ax.plot(loss_sum[1], color='r', alpha=0.2)
    ax.errorbar(test_epochs,test_metrics[0],
                yerr=test_metrics[1], label='test ssim')
    ax.errorbar(test_epochs,test_metrics[4],
                yerr=test_metrics[5], label='test mse')
    ax.set_yscale('log')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.title.set_text('pSNR (over validation set)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('pSNR [dB]')
    ax.errorbar(test_epochs,test_metrics[2],
                yerr=test_metrics[3])
    plt.savefig(fname)
    plt.close()
    print('saved image at: %s'%fname)

def main(args):
    mpl.rcParams.update({'font.size': 22})
    
    model = eval('get_' + args.model)(args)
    model = MyDataParallel(model, device_ids=args.dev_ids)
    model = model.to(args.device)
    model.eval()

    #loading model
    fname = os.path.join(args.dir_final_test, 'best_model.txt')
    with open(fname, 'r') as f:
        lname = f.read()
        f.close()
    model.load_state_dict(torch.load(lname))

    start = tm.time()
    make_plots(args)
    metrics = inference(args, model)
    print('Final test time: %.4f\n'%(tm.time()-start))
    print('Final test ssim: %.4f +/- %.4f'%(metrics[0], metrics[1]))
    print('Final test psnr: %.4f +/- %.4f'%(metrics[2], metrics[3]))
    print('Final test mse: %.4f +/- %.4f'%(metrics[4], metrics[5]))
    
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
    args['loss_fn'] = "_".join(["loss", args['loss_fn']])
    print('Working on device: {}\n'.format(args['device']))
    args = Args(**args)
    start = tm.time()
    main(args)
    print('Program done in %f'%(tm.time()-start))