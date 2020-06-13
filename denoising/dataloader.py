import os
import torch
import numpy as np

from model_utils import plot_crops
from model_utils import plot_wires


class CropLoader(torch.utils.data.Dataset):
    def __init__(self, args, name):
        data_dir = args.dataset_dir
        patch_size = args.crop_size[0]
        p = args.crop_p

        fname = os.path.join(data_dir,
                             'clear_crops/readout_%s_%d_%f'%(name,
                                                                patch_size,
                                                                p))
        readout_clear = torch.Tensor(np.load(fname))
        if args.plot_dataset:
            sample = torch.randint(0,readout_clear.shape[0],(25,))
            wire = torch.randint(0,patch_size, (25,))
            plot_crops(args.dir_testing,
                       readout_clear,
                       "_".join(["readout_clear",name]),sample)
            plot_wires(args.dir_testing,
                       readout_clear,
                       "_".join(["readout_clear",name]),sample,wire)

        fname = os.path.join(data_dir,
                             'noised_crops/readout_%s_%d_%f'%(name,
                                                                patch_size,
                                                                p))
        readout_noise = torch.Tensor(np.load(fname))
        if args.plot_dataset:
            plot_crops(args.dir_testing,
                       readout_noise,
                       "_".join(["readout_noisy",name]),sample)
            plot_wires(args.dir_testing,
                       readout_noise,
                       "_".join(["readout_noisy",name]),sample,wire)
        

        fname = os.path.join(data_dir,
                             'clear_crops/collection_%s_%d_%f'%(name,
                                                                patch_size,
                                                                p))
        collection_clear = torch.Tensor(np.load(fname))
        if args.plot_dataset:
            sample = torch.randint(0,collection_clear.shape[0],(25,))
            wire = torch.randint(0,patch_size, (25,))
            plot_crops(args.dir_testing,
                       collection_clear,
                       "_".join(["collection_clear",name]),sample)
            plot_wires(args.dir_testing,
                       collection_clear,
                       "_".join(["collection_clear",name]),sample, wire)

        

        fname = os.path.join(data_dir,
                             'noised_crops/collection_%s_%d_%f'%(name,
                                                                patch_size,
                                                                p))
        collection_noise = torch.Tensor(np.load(fname))
        if args.plot_dataset:
            plot_crops(args.dir_testing,
                       collection_noise,
                       "_".join(["collection_noisy",name]),sample)
            plot_wires(args.dir_testing,
                       collection_noise,
                       "_".join(["collection_noisy",name]),sample,wire)

        self.clear_crops = torch.cat([collection_clear, readout_clear])
        self.noised_crops = torch.cat([collection_noise, readout_noise])

        #self.clear_crops = collection_clear
        #self.noised_crops = collection_noise

        self.clear_crops = self.clear_crops.unsqueeze(1)
        self.noised_crops = self.noised_crops.unsqueeze(1)
     
    def __len__(self):
        return len(self.noised_crops)
    def __getitem__(self, index):
        return self.clear_crops[index], self.noised_crops[index]

class PlaneLoader(torch.utils.data.Dataset):
    def __init__(self, args, file):
        data_dir = args.dataset_dir

        fname = os.path.join(data_dir, 'clear_planes/%s'%file)
        self.clear_planes = torch.Tensor(np.load(fname)).unsqueeze(1)

        if args.plot_dataset:
            sample = torch.randint(0,self.clear_planes.shape[0],(25,))
            wire = torch.randint(0,self.clear_planes.shape[2], (25,))

            plot_wires(args.dir_testing,
                       self.clear_planes[:,0],
                       "_".join([file, "clear"]),sample,wire)

        fname = os.path.join(data_dir, 'noised_planes/%s'%file)
        self.noised_planes = torch.Tensor(np.load(fname)).unsqueeze(1)
        if args.plot_dataset:
            plot_wires(args.dir_testing,
                       self.noised_planes[:,0],
                       "_".join([file, "noisy"]),sample,wire)
     
    def __len__(self):
        return len(self.noised_planes)
    def __getitem__(self, index):
        return self.clear_planes[index], self.noised_planes[index]
