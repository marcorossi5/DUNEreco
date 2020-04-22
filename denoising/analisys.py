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
parser.add_argument("--device", "-d", default="-1", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")

def inference(args, model):
    #load dataset
    test_data = [torch.utils.data.DataLoader(PlaneLoader(args.dataset_dir,
                                                      'readout_test'
                                                      ),
                                        num_workers=args.num_workers),
                 torch.utils.data.DataLoader(PlaneLoader(args.dataset_dir,
                                                      'readout_test'
                                                      ),
                                        num_workers=args.num_workers)]

    #test_data = [load_planes(args.dataset_dir, 'collection_test'),
    #             load_planes(args.dataset_dir, 'readout_test')]
    legend = ['collection', 'readout']

    mse_loss = torch.nn.MSELoss()
    psnr = []
    mse = []
    res = [[],[]]
    labels = [[],[]]
    p_x, p_y = model.patch_size
    split_size = 256
    #print('Number of planes to be tested:', len(test_data))
    for i, data in enumerate(test_data):
        for (clear, noised) in data:
            labels[i] += [clear]
            
            crops, crops_shape, pad = split_img(noised, (p_x,p_y))
            loader = torch.split(crops, split_size)
            dn = []
            for chunk in loader:
                answer = model(chunk.to(device)).cpu().data
                dn.append(answer)

            dn = torch.cat(dn)

            dn = recombine_img(dn, crops_shape, pad)

            res[i] += [dn]

            #res[i] += [model.forward_image(noised,
            #                               args.device,
            #                               args.test_batch_size)]
            psnr.append(compute_psnr(clear, res[i][-1]))
            mse.append(mse_loss(clear, res[i][-1]).item())
        labels[i] = np.concatenate(labels[i])[:,0]
        res[i] = np.concatenate(res[i])[:,0]
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

    ax = fig.add_subplot(311)
    ax.title.set_text('Sample of Clear image')
    z = ax.imshow(labels[0][0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(312)
    ax.title.set_text('Sample of |Denoised - Clear|')
    z = ax.imshow(diff[0][0])
    fig.colorbar(z, ax=ax)

    ax = fig.add_subplot(325)
    ax.hist([i[0].flatten() for i in diff], 100,
             stacked=True, label=legend,
             density=True, histtype='step')
    ax.set_yscale('log')
    ax.legend()
    ax.title.set_text('Sample of histogram of |Diff|')

    ax = fig.add_subplot(326)
    ax.hist([i.flatten() for i in diff], 100,
             stacked=True, label=legend,
             density=True, histtype='step')
    ax.set_yscale('log')
    ax.legend()
    ax.title.set_text('Histogram of all |Diff|')

    plt.savefig(fname)
    plt.close()

    return np.array([np.mean(psnr),
                    np.std(psnr)/np.sqrt(len(psnr)),
                    np.mean(mse),
                    np.std(mse)/np.sqrt(len(psnr))])

def make_plots(args):
    fname = os.path.join(args.dir_metrics, 'loss_sum.npy')
    loss_sum = np.load(fname)

    #smoothing the loss
    weight = 0.8
    #weight = 2/(len(loss_sum)+1)
    loss_avg = moving_average(loss_sum, weight)

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
    ax.plot(loss_avg, color='#ff7f0e', label='train loss')
    ax.plot(loss_sum, color='#ff7f0e', alpha=0.2)
    ax.errorbar(test_epochs,test_metrics[2],
                yerr=test_metrics[3], label='test loss')
    ax.set_yscale('log')
    ax.legend()

    ax = fig.add_subplot(122)
    ax.title.set_text('pSNR (over validation set)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('pSNR [dB]')
    ax.errorbar(test_epochs,test_metrics[0],
                yerr=test_metrics[1])
    plt.savefig(fname)
    plt.close()
    print('saved image at: %s'%fname)

def main(args):
    mpl.rcParams.update({'font.size': 22})
    
    model = eval('get_' + args.model)(args.k,
                                args.in_channels,
                                args.hidden_channels,
                                args.crop_size
                                )
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
    metrics = inference(args, model)
    print('Final test time: %.4f\n'%(tm.time()-start))
    print('Final test psnr: %.4f +/- %.4f'%(metrics[0], metrics[1]))
    print('Final test loss: %.4f +/- %.4f'%(metrics[2], metrics[3]))
    
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