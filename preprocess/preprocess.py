import os
import sys
import argparse
import time
import numpy as np
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", default="*",
                    type=str, help='Source .npy file name')
parser.add_argument("--dir_name", "-p", default="../datasets",
                    type=str, help='Directory path to datasets')
parser.add_argument("--n_crops", "-n", default=500, type=int,
                    help="number of crops for each plane")
parser.add_argument("--crop_edge", "-c", default=32, type=int,
                    help="crop shape")
parser.add_argument("--percentage", "-x", default=0.5, type=float,
                    help="percentage of signal")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing_utils as putils
from utils.utils import get_freer_gpu
from denoising.ssim import _fspecial_gauss_1d, stat_gaussian_filter

#crop_shape = (32,32)

def get_planes_and_dump(source, dir_name):
    path_clear = os.path.join(dir_name,"clear_events", source)
    path_noise = os.path.join(dir_name,"noised_events", source)

    clear_readout_planes = []
    noised_readout_planes = []
    clear_collection_planes = []
    noised_collection_planes = []

    for file_clear, file_noise in putils.load_files(path_clear, path_noise):
        a, b, c, d = putils.get_planes(file_clear, file_noise)
        clear_readout_planes.append(a)
        noised_readout_planes.append(b)
        clear_collection_planes.append(c)
        noised_collection_planes.append(d)

    clear_readout_planes = np.concatenate(clear_readout_planes)
    noised_readout_planes = np.concatenate(noised_readout_planes)
    clear_collection_planes = np.concatenate(clear_collection_planes)
    noised_collection_planes = np.concatenate(noised_collection_planes)

    s = len(clear_readout_planes)
    p = np.random.permutation(s)
    c_r = clear_readout_planes[p]
    n_r = noised_readout_planes[p]

    np.save(os.path.join(dir_name,"clear_planes", 'readout_train'),
            c_r[:int(s*0.6)])
    np.save(os.path.join(dir_name,"clear_planes", 'readout_val'),
            c_r[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"clear_planes", 'readout_test'),
            c_r[int(s*0.8):])

    np.save(os.path.join(dir_name,"noised_planes", 'readout_train'),
            n_r[:int(s*0.6)])
    np.save(os.path.join(dir_name,"noised_planes", 'readout_val'),
            n_r[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"noised_planes", 'readout_test'),
            n_r[int(s*0.8):])

    s = len(clear_collection_planes)
    p = np.random.permutation(s)
    c_c = clear_collection_planes[p]
    n_c = noised_collection_planes[p]

    np.save(os.path.join(dir_name,"clear_planes", 'collection_train'),
            c_c[:int(s*0.6)])
    np.save(os.path.join(dir_name,"clear_planes", 'collection_val'),
            c_c[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"clear_planes", 'collection_test'),
            c_c[int(s*0.8):])

    np.save(os.path.join(dir_name,"noised_planes", 'collection_train'),
            n_c[:int(s*0.6)])
    np.save(os.path.join(dir_name,"noised_planes", 'collection_val'),
            n_c[int(s*0.6):int(s*0.8)])
    np.save(os.path.join(dir_name,"noised_planes", 'collection_test'),
            n_c[int(s*0.8):])

def crop_planes_and_dump(dir_name, n_crops, crop_shape, p):
    for s in ['readout_', 'collection_']:
        for ss in ['train', 'val', 'test']:
            #create channel dimension
            clear_planes = np.load(os.path.join(dir_name,
                                                 "clear_planes",
                                                 s+ss+".npy"))[:,None]
            noised_planes = np.load(os.path.join(dir_name,
                                                 "noised_planes",
                                                 s+ss+".npy"))[:,None]
            clear_min,\
            clear_max = putils.get_normalization(clear_planes)
            
            noised_min,\
            noised_max = putils.get_normalization(noised_planes)

            win = _fspecial_gauss_1d(17,4).unsqueeze(1)
            filt_1 = stat_gaussian_filter(torch.Tensor(noised_planes).to(0),
                                                win.to(0)).cpu()

            win = _fspecial_gauss_1d(101,32).unsqueeze(1)
            filt_2 = stat_gaussian_filter(torch.Tensor(noised_planes).to(0),
                                                win.to(0)).cpu()

            noised_planes = np.concatenate([noised_planes,
                                            np.array(filt_1),
                                            np.array(filt_2),],1)
            
            clear_crops = []
            noised_crops = []
            for clear_plane, noised_plane in zip(clear_planes,noised_planes):

                idx = putils.get_crop(clear_plane[0],
                                      n_crops = n_crops,
                                      crop_shape = crop_shape,
                                      p = p)

                idx_clear = np.zeros([n_crops,1,1,1]).astype(int)
                idx_noisy = np.arange(3, dtype=np.int32).reshape(1,-1,1,1)
                idx_noisy = np.repeat(idx_noisy, n_crops,0)

                clear_crops.append(clear_plane[tuple([idx_clear]+idx)])
                noised_crops.append(noised_plane[tuple([idx_noisy]+idx)])

            clear_crops = np.concatenate(clear_crops, 0)
            noised_crops = np.concatenate(noised_crops, 0)
                
            np.save(os.path.join(dir_name,
                                 "clear_crops",
                                 "%s%s_%d_%f"%(s, ss, crop_shape[0], p)),
                    (clear_crops-clear_min)/(clear_max-clear_min))

            
            np.save(os.path.join(dir_name,
                                "noised_crops",
                                "%s%s_%d_%f"%(s, ss, crop_shape[0], p)),
                    (noised_crops-noised_min)/(noised_max-noised_min))

            np.save(os.path.join(dir_name,
                                "clear_crops",
                                "postprocess_%s%s_%d_%f"%(s, ss, crop_shape[0], p)),
                    np.array([clear_min, clear_max])
                    )

            np.save(os.path.join(dir_name,
                                "noised_crops",
                                "postprocess_%s%s_%d_%f"%(s, ss, crop_shape[0], p)),
                    np.array([noised_min, noised_max])
                    )
            
def main(source, dir_name, n_crops, crop_edge, percentage):
    crop_shape = (crop_edge, crop_edge)
    for i in ['clear_planes', 'clear_crops', 'noised_planes', 'noised_crops']:
        if not os.path.isdir(os.path.join(dir_name,i)):
            os.mkdir(os.path.join(dir_name,i))

    get_planes_and_dump(source, dir_name)
    for s in ['readout_', 'collection_']:
        for ss in ['train', 'val', 'test']:
            print(s+ss + ' clear planes',
                  np.load(os.path.join(dir_name,
                                       'clear_planes', s+ss+".npy")).shape)
            print(s+ss + ' noised planes',
                  np.load(os.path.join(dir_name,
                                       'noised_planes', s+ss+".npy")).shape)
    
    crop_planes_and_dump(dir_name, n_crops, crop_shape, percentage)
    for s in ['readout_', 'collection_']:
        for ss in ['train', 'val', 'test']:
            print(s+ss + ' clear crops',
                  np.load(os.path.join(dir_name,
                                       'clear_crops',
                                       "%s%s_%d_%f.npy"%(s, ss,
                                                     crop_shape[0],
                                                     percentage))).shape)
            print(s+ss + ' noised crops',
                  np.load(os.path.join(dir_name,
                                       'noised_crops',
                                       "%s%s_%d_%f.npy"%(s, ss,
                                                     crop_shape[0],
                                                     percentage))).shape)

if __name__ == '__main__':
    args = vars(parser.parse_args())

    print("Args:")
    for k  in args.keys():
      print(k, args[k])


    start = time.time()
    main(**args)
    print('Program done in %f'%(time.time()-start))