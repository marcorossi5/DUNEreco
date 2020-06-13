import os
import ssim
import numpy as np
import torch
import matplotlib.pyplot as plt

from args import Args

from ssim import stat_ssim

def main(args):
    """Main function: plots SSIM of a batch of crops to select k1, k2 parameters"""
    fname = os.path.join(data_dir,
                             'clear_crops/collection_val_32_0.500000.npy')
    clear = torch.Tensor(np.load(fname)[:2048]).unsqueeze(1)


    fname = os.path.join(data_dir,
                             'noised_crops/collection_val_32_0.500000.npy')
    noisy = torch.Tensor(np.load(fname)[:2048]).unsqueeze(1)

    print("Number of crops: ", len(clear))

    y = []
    x = np.logspace(-5,-1,10)

    for i in x:
        y.append(stat_ssim(noisy, clear,
                           data_range=1., size_average=True),
                           k=(i,i))

    plt.plot(x,y)
    plt.savefig("../collection_t.png")

if __name__ == '__main__':
    ARGS = Args(**ARGS)
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
