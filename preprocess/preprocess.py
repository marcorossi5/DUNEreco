import os
import sys
import argparse
import time
import glob
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--dir_name", "-p", default="../datasets/denoising",
                    type=str, help='Directory path to datasets')
parser.add_argument("--n_crops", "-n", default=1000, type=int,
                    help="number of crops for each plane")
parser.add_argument("--crop_edge", "-c", default=32, type=int,
                    help="crop shape")
parser.add_argument("--percentage", "-x", default=0.5, type=float,
                    help="percentage of signal")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import putils
from utils.utils import get_freer_gpu
from denoising.ssim import _fspecial_gauss_1d, stat_gaussian_filter

#crop_shape = (32,32)
APAs = 6
r_step = 800
c_step = 960
rc_step = 2*r_step + c_step

def get_planes_and_dump(dname):
    readout_clear = []
    readout_noisy = []
    collection_clear = []
    collection_noisy = []

    path_clear = glob.glob(f'{dname}/evts/*noiseoff*')
    path_noisy = glob.glob(f'{dname}/evts/*noiseon*')

    for file_clear, file_noisy in zip(path_clear, path_noisy):
        c = np.load(file_clear)[:,2:]
        n = np.load(file_noisy)[:,2:]
        for i in range(APAs):
            readout_clear.append(c[i*rc_step:i*rc_step+r_step])
            readout_noisy.append(n[i*rc_step:i*rc_step+r_step])
            readout_clear.append(c[i*rc_step+r_step:i*rc_step+2*r_step])
            readout_noisy.append(n[i*rc_step+r_step:i*rc_step+2*r_step])
            collection_clear.append(c[i*rc_step+2*r_step:(i+1)*rc_step])
            collection_noisy.append(n[i*rc_step+2*r_step:(i+1)*rc_step])
    
    collection_clear = np.stack(collection_clear,0)[:,None]
    collection_noisy = np.stack(collection_noisy,0)[:,None]
    readout_clear = np.stack(readout_clear,0)[:,None]
    readout_noisy = np.stack(readout_noisy,0)[:,None]

    print("\tCollection clear planes: ", collection_clear.shape)
    print("\tCollection noisy planes: ", collection_noisy.shape)
    print("\tReadout clear planes: ", readout_clear.shape)
    print("\tReadout noisy planes: ", readout_noisy.shape)

    fname = os.path.join(dname, "planes", "readout_clear")
    np.save(fname,
            np.stack(readout_clear,0))

    fname = os.path.join(dname, "planes", "readout_noisy")
    np.save(fname,
            np.stack(readout_noisy,0))

    fname = os.path.join(dname, "planes", "collection_clear")
    np.save(fname,
            np.stack(collection_clear,0))

    fname = os.path.join(dname, "planes", "collection_noisy")
    np.save(fname,
            np.stack(collection_noisy,0))

def crop_planes_and_dump(dir_name, n_crops, crop_shape, p):
    for s in ['readout', 'collection']:
        fname = os.path.join(dir_name,"planes",f'{s}_clear.npy')
        clear_planes = np.load(fname)[:,0]

        fname = os.path.join(dir_name,"planes",f'{s}_noisy.npy')
        noisy_planes = np.load(fname)[:,0]

        clear_m = clear_planes.min()
        clear_M = clear_planes.max()
        noisy_m = noisy_planes.min()
        noisy_M = noisy_planes.max()

        clear_crops = []
        noisy_crops = []
        for clear_plane, noisy_plane in zip(clear_planes,noisy_planes):
            idx = putils.get_crop(clear_plane,
                                  n_crops = n_crops,
                                  crop_shape = crop_shape,
                                  p = p)
            clear_crops.append(clear_plane[idx])
            noisy_crops.append(noisy_plane[idx])

        clear_crops = np.concatenate(clear_crops, 0)[:,None]
        noisy_crops = np.concatenate(noisy_crops, 0)[:,None]

        print(f'\n{s} clear crops:', clear_crops.shape)
        print(f'{s} noisy crops:', noisy_crops.shape)

        fname = os.path.join(dir_name,
                             f'{s}_clear_{crop_shape[0]}_{p}')
        np.save(fname,
                (clear_crops-clear_m)/(clear_M-clear_m))

        fname = os.path.join(dir_name,
                             f'{s}_noisy_{crop_shape[0]}_{p}')
        np.save(fname,
                (noisy_crops-noisy_m)/(noisy_M-noisy_m))
            
def main(dir_name, n_crops, crop_edge, percentage):
    crop_shape = (crop_edge, crop_edge)
    for i in ['train/crops', 'train/planes', 'val/planes', 'test/planes']:
        if not os.path.isdir(os.path.join(dir_name,i)):
            os.mkdir(os.path.join(dir_name,i))

    for s in ['train', 'test', 'val']:
        print(f'\n{s}:')
        dname = os.path.join(dir_name, s)
        get_planes_and_dump(dname)

    dname = os.path.join(dir_name, 'train')
    crop_planes_and_dump(dname, n_crops, crop_shape, percentage)
    
if __name__ == '__main__':
    args = vars(parser.parse_args())

    print("Args:")
    for k  in args.keys():
      print(k, args[k])


    start = time.time()
    main(**args)
    print('Program done in %f'%(time.time()-start))