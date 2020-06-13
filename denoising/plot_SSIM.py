import os
import ssim
import numpy as np
import torch
import matplotlib.pyplot as plt

from args import Args

from ssim import ssim_stat

def main(args):
    """Main function: plots SSIM of a batch of crops to select k1, k2 parameters"""
    fname = os.path.join(data_dir,
                             'clear_crops/collection_%s_%d_%f.npy'%(name,
                                                                patch_size,
                                                                p))
    clear = torch.Tensor(np.load(fname))


    fname = os.path.join(data_dir,
                             'noised_crops/collection_%s_%d_%f.npy'%(name,
                                                                patch_size,
                                                                p))
    noisy = torch.Tensor(np.load(fname))

if __name__ == '__main__':
    ARGS = Args(**ARGS)
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
