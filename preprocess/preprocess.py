import numpy as np
import torch
import os, sys
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", default="*", type=str, help='Source .npy file name inside clear(noised)_files directory')
parser.add_argument("--dir_name", "-p", default="../datasets", type=str, help='Directory path to datasets')
parser.add_argument("--device", "-d", default="-1", type=str, help="-1 (automatic)/ -2 (cpu) / gpu number")
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import preprocessing_utils as putils
from utils.utils import get_freer_gpu

crop_shape = (32,32)
n_crops = 500
n_max = 10 #max parallelization in 16GB memory

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

    torch.save(torch.Tensor(c_r[:int(s*0.6)]), os.path.join(dir_name,"clear_planes", 'readout_train'))
    torch.save(torch.Tensor(c_r[int(s*0.6):int(s*0.8)]), os.path.join(dir_name,"clear_planes", 'readout_val'))
    torch.save(torch.Tensor(c_r[int(s*0.8):]), os.path.join(dir_name,"clear_planes", 'readout_test'))

    torch.save(torch.Tensor(n_r[:int(s*0.6)]), os.path.join(dir_name,"noised_planes", 'readout_train'))
    torch.save(torch.Tensor(n_r[int(s*0.6):int(s*0.8)]), os.path.join(dir_name,"noised_planes", 'readout_val'))
    torch.save(torch.Tensor(n_r[int(s*0.8):]), os.path.join(dir_name,"noised_planes", 'readout_test'))


    s = len(clear_collection_planes)
    p = np.random.permutation(s)
    c_c = clear_collection_planes[p]
    n_c = noised_collection_planes[p]

    torch.save(torch.Tensor(c_c[:int(s*0.6)]), os.path.join(dir_name,"clear_planes", 'collection_train'))
    torch.save(torch.Tensor(c_c[int(s*0.6):int(s*0.8)]), os.path.join(dir_name,"clear_planes", 'collection_val'))
    torch.save(torch.Tensor(c_c[int(s*0.8):]), os.path.join(dir_name,"clear_planes", 'collection_test'))

    torch.save(torch.Tensor(n_c[:int(s*0.6)]), os.path.join(dir_name,"noised_planes", 'collection_train'))
    torch.save(torch.Tensor(n_c[int(s*0.6):int(s*0.8)]), os.path.join(dir_name,"noised_planes", 'collection_val'))
    torch.save(torch.Tensor(n_c[int(s*0.8):]), os.path.join(dir_name,"noised_planes", 'collection_test'))

def crop_planes_and_dump(dir_name, device):
    for s in ['readout_', 'collection_']:
        for ss in ['train', 'val', 'test']:
            
            clear_planes = torch.load(os.path.join(dir_name,"clear_planes", s+ss))

            idx_b, idx_c = putils.get_crop(clear_planes,total_crops=n_crops,crop_shape=crop_shape,device=device, n_max=n_max)
            
            n,_,_  = clear_planes.shape
            idx_a = torch.arange(n).repeat(n_crops).reshape(-1,1,1)
            idx = (idx_a, idx_b, idx_c)
            torch.save(clear_planes[idx], os.path.join(dir_name,"clear_crops", s+ss))
            del clear_planes
            noised_planes = torch.load(os.path.join(dir_name,"noised_planes", s+ss))
            torch.save(noised_planes[idx], os.path.join(dir_name,"noised_crops", s+ss))
            del noised_planes
            


def main(source, dir_name, device):
    
    for i in ['clear_planes', 'clear_crops', 'noised_planes', 'noised_crops']:
        if not os.path.isdir(os.path.join(dir_name,i)):
            os.mkdir(os.path.join(dir_name,i))

    get_planes_and_dump(source, dir_name)
    for s in ['readout_', 'collection_']:
        for ss in ['train', 'val', 'test']:
            print(s+ss + ' clear', torch.load(os.path.join(dir_name, 'clear_planes', s+ss)).shape)
            print(s+ss + ' noised', torch.load(os.path.join(dir_name, 'noised_planes', s+ss)).shape)
    

    
    crop_planes_and_dump(dir_name,device)
    for s in ['readout_', 'collection_']:
        for ss in ['train', 'val', 'test']:
            print(s+ss + ' clear', torch.load(os.path.join(dir_name, 'clear_crops', s+ss)).shape)
            print(s+ss + ' noised', torch.load(os.path.join(dir_name, 'noised_crops', s+ss)).shape)

    

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
    start = time.time()
    main(**args)
    print('Program done in %f'%(time.time()-start))
