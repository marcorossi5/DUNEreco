import os
import torch
import numpy as np

from model_utils import plot_crops
from model_utils import plot_wires

from ssim import _fspecial_gauss_1d, stat_gaussian_filter


class CropLoader(torch.utils.data.Dataset):
    def __init__(self, args, folder, channel):
        data_dir = args.dataset_dir
        patch_size = args.crop_size[0]
        p = args.crop_p

        fname = os.path.join(data_dir,'train','crops',
                             f'{channel}_clear_{patch_size}_{p}.npy')
        self.clear = torch.Tensor(np.load(fname))

        fname = os.path.join(data_dir,'train','crops',
                             f'{channel}_noisy_{patch_size}_{p}.npy')
        self.noisy = torch.Tensor(np.load(fname))

        if args.plot_dataset:
            sample = torch.randint(0,self.clear.shape[0],(25,))
            wire = torch.randint(0,patch_size, (25,))
            plot_crops(args.dir_testing,
                       clear[:,0],
                       "_".join([channel,'clear',folder]),sample)
            plot_wires(args.dir_testing,
                       clear[:,0],
                       "_".join([channel,'clear',folder]),sample, wire)
            plot_crops(args.dir_testing,
                       noisy[:,0],
                       "_".join([channel,'noisy',folder]),sample)
            plot_wires(args.dir_testing,
                       noisy[:,0],
                       "_".join([channel,'noisy',folder]),sample,wire)
    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, index):
        return self.clear[index], self.noisy[index]

class PlaneLoader(torch.utils.data.Dataset):
    def __init__(self, args, folder, channel):
        data_dir = os.path.join(args.dataset_dir, folder)

        fname = os.path.join(data_dir, 'planes', f'{channel}_clear.npy')
        self.clear = torch.Tensor(np.load(fname))
        
        fname = os.path.join(data_dir, 'planes', f'{channel}_noisy.npy')
        self.noisy = torch.Tensor(np.load(fname))
        
        if args.plot_dataset:
            sample = torch.randint(0,self.clear.shape[0],(25,))
            wire = torch.randint(0,self.clear.shape[2], (25,))
            plot_wires(args.dir_testing,
                       self.clear[:,0],
                       "_".join([folder, file, "clear"]),sample,wire)
            plot_wires(args.dir_testing,
                       self.noisy[:,0],
                       "_".join([folder, file, "noisy"]),sample,wire)
        
        clear_M = self.clear.max()
        clear_m = self.clear.min()

        noisy_M = self.noisy.max()
        noisy_m = self.noisy.min()


        self.clear = (self.clear-clear_m)/(clear_M-clear_m)
        self.noisy = (self.noisy-noisy_m)/(noisy_M-noisy_m)

        self.norm = torch.stack([clear_M, clear_m, noisy_M, noisy_m])

    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, index):
        return self.clear[index], self.noisy[index], self.norm
