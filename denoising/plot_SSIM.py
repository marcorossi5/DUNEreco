import os
import ssim
import numpy as np
import torch
import matplotlib.pyplot as plt

from args import Args

from ssim import stat_ssim

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir_name", "-p", default="../datasets",
                    type=str, help='Directory path to datasets')
PARSER.add_argument("--device", "-d", default="0", type=str,
                    help="-1 (automatic)/ -2 (cpu) / gpu number")

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
    print('Working on device: {}\n'.format(ARGS['device']))
    ARGS['epoch'] = None
    ARGS['model'] = None
    ARGS = Args(**ARGS)
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
