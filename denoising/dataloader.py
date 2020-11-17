import os
import torch
import numpy as np

from model_utils import plot_crops
from model_utils import plot_wires
from model_utils import calculate_pad
from model_utils import Converter

from ssim import _fspecial_gauss_1d, stat_gaussian_filter


class CropLoader(torch.utils.data.Dataset):
    def __init__(self, args, folder, channel):
        """
        This function loads the crops for training.
        Crops are normalized in the [0,1] range with the minmax
        normalization\.
        Parameters:
            args: Args object
            folder: str, one of ['train','val','test']
            channel: str, one of ['readout','collection']
        """
        data_dir = args.dataset_dir
        patch_size = args.patch_size[0]
        p = args.crop_p

        fname = os.path.join(data_dir,'train','crops',
                             f'{channel}_clear_{patch_size}_{p}.npy')
        clear = torch.Tensor(np.load(fname))

        fname = os.path.join(data_dir,'train','crops',
                             f'{channel}_noisy_{patch_size}_{p}.npy')
        noisy = torch.Tensor(np.load(fname))

        #normalize crops
        fname = os.path.join(args.dataset_dir,
                             '_'.join([channel,'normalization.npy']))
        m, M = np.load(fname)

        hits = torch.clone(clear)
        hits[hits!=0] = 1

        self.clear = (clear-m)/(M-m)
        self.noisy = (noisy-m)/(M-m)

        self.clear = torch.cat([self.clear, hits],1)

    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, index):
        return self.clear[index], self.noisy[index]

class PlaneLoader(torch.utils.data.Dataset):
    def __init__(self, args, folder, channel):
        """
        This function loads the planes for inference.
        Only noisy planes are normalized since clear planes don't
        need to be scaled at inference time.
        Parameters:
            args: Args object
            folder: str, one of ['train','val','test']
            channel: str, one of ['readout','collection']
            t: float, threshold to be put on labels
        """
        self.patch_size = args.patch_size
        data_dir = os.path.join(args.dataset_dir, folder)

        fname = os.path.join(data_dir, 'planes', f'{channel}_noisy.npy')
        noisy = torch.Tensor(np.load(fname))
        
        fname = os.path.join(args.dataset_dir,
                             '_'.join([channel,'normalization.npy']))
        self.norm = np.load(fname)

        self.noisy = (noisy-self.norm[0])/(self.norm[1]-self.norm[0])
        self.converter = Converter(self.patch_size, self.norm)
        self.splits = self.converter.planes2tiles(self.noisy)

        fname = os.path.join(data_dir, 'planes', f'{channel}_clear.npy')
        clear = torch.Tensor(np.load(fname))

        hits = torch.clone(clear)
        hits[hits!= 0] = 1
        self.clear = torch.cat([clear, hits],1)

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        return self.splits[index]

# TODO: is the label generation in the PlaneLoader correct according to the
# threshold considerations?
