""" This module computes the Wiener filter for planes in the test set"""
import sys
import os
import argparse
import numpy as np
import time as tm
from numpy.fft import fft2, ifft2
from scipy.signal import gaussian, convolve2d
import matplotlib.pyplot as plt
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from losses import loss_mse, loss_ssim

from utils.utils import compute_psnr

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--device", "-d", default="0", type=str,
                    help="gpu number")
PARSER.add_argument("--kernel_size", "-k", default=3, type=int,
                    help="size of the gaussian filter")

def wiener_filter(img, kernel, K):
    kernel /= np.sum(kernel)
    dummy = np.copy(img)
    dummy = fft2(dummy)
    kernel = fft2(kernel, s= img.shape)
    kernel = np.conj(kernel) / (np.abs(kernel) ** 2 + K)
    dummy = dummy * kernel
    dummy = np.abs(ifft2(dummy))
    return dummy

def gaussian_kernel(kernel_size = 3):
    # parameters:
    # kernel size is the filter size
    # sigma of the gaussian (always zero mean)
    h = gaussian(kernel_size, kernel_size / 3).reshape(kernel_size, 1)
    # make it a 2d filter
    h = np.dot(h, h.transpose())
    # normalize
    h /= np.sum(h)
    return h

def plot():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=3, ncols=1, wspace=3)

    ax = fig.add_subplot(gs[0])
    ax.plot(img[10], lw=0.3)
    #z = ax.imshow(img)
    #plt.colorbar(z, ax=ax)
    ax.set_title('Labels')

    ax = fig.add_subplot(gs[1])
    ax.plot(filtered_img[10], lw=0.3)
    #z = ax.imshow(filtered_img)
    #plt.colorbar(z, ax=ax)
    ax.set_title('Predicted')
        
    plt.savefig('wiener_filter.png', dpi=300, bbox_inches='tight')
    plt.close()


def main(device, kernel_size):
    # Load target images
    file_name = os.path.join("../datasets/denoising/val", 'planes', 'collection_clear.npy') 
    img = np.load(file_name)

    # Load noisy images
    file_name = os.path.join("../datasets/denoising/val", 'planes', 'collection_noisy.npy') 
    
    # subtract the pedestal already
    # pedestal = 500
    noisy_img = np.load(file_name) - 500

    # Apply Wiener Filter
    out_img = []
    kernel = gaussian_kernel(kernel_size)
    for i in noisy_img[:,0]:
        out_img.append(wiener_filter(i, kernel, K = 10))

    img = torch.Tensor(img).to(device)
    out_img = np.stack(out_img)[:,None]
    out_img = torch.Tensor(out_img).to(device)

    ssim = []
    mse = []
    psnr = []

    for i,j in zip(img, out_img):
        ssim.append(1-loss_ssim()(img, out_img).cpu().item())
        mse.append(loss_mse()(img, out_img).cpu().item())
        psnr.append(compute_psnr(img, out_img))

    ssim_mean = np.mean(ssim)
    ssim_std = np.mean(ssim) / np.sqrt(len(ssim))

    mse_mean = np.mean(mse)
    mse_std = np.mean(mse) / np.sqrt(len(ssim))

    psnr_mean = np.mean(psnr)
    psnr_std = np.mean(psnr) / np.sqrt(len(ssim))

    res = np.array([[ssim_mean, ssim_std],
                    [mse_mean, mse_std],
                    [psnr_mean, psnr_std]])
    fname = f'./denoising/benchmarks/results/wiener_{kernel_size}_metrics'
    np.save(fname, res)


if __name__ == '__main__':
    ARGS = vars(PARSER.parse_args())
    gpu = torch.cuda.is_available()
    dev = ARGS['device']
    dev = f'cuda:{dev}' if gpu else 'cpu'
    ARGS['device'] = torch.device(dev)
    START = tm.time()
    main(**ARGS)
    print('Program done in %f'%(tm.time()-START))
    
