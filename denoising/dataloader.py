import os
import torch
import numpy as np

from model_utils import plot_crops
from model_utils import plot_wires
from model_utils import calculate_pad
from model_utils import Converter

from ssim import _fspecial_gauss_1d, stat_gaussian_filter


class CropLoader(torch.utils.data.Dataset):
    def __init__(self, args):
        """
        This function loads the crops for training.
        Crops are normalized in the [0,1] range with the minmax
        normalization\.
        Parameters:
            args: Args object
        """
        data_dir = args.dataset_dir
        edge_patch = args.patch_size[0]
        p = args.crop_p
        fname = os.path.join(data_dir,'train/crops',
                             f"{args.channel}_noisy_{edge_patch}_{p}.npy")
        self.noisy = torch.Tensor( np.load(fname) )
        fname = os.path.join(data_dir,'train/crops',
                         f'{args.channel}_clear_{edge_patch}_{p}.npy')
        clear = torch.Tensor( np.load(fname) )
        hits = torch.clone(clear)
        mask = (hits <= args.threshold) & (hits >= -args.threshold)
        hits[mask] = 0
        hits[~mask] = 1
        self.balance_ratio = np.count_nonzero(hits)/hits.numel()
        self.clear = torch.cat([clear, hits],1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.clear[index], self.noisy[index]

class PlaneLoader(torch.utils.data.Dataset):
    def __init__(self, dataset_dir, folder, task, channel, threshold):
        """
        This function loads the planes for inference.
        Only noisy planes are normalized since clear planes don't
        need to be scaled at inference time.
        Parameters:
            args: Args object
            folder: str, one of ['train','val','test']
            t: float, threshold to be put on labels
        """
        data_dir = os.path.join(dataset_dir, folder)
        fname = os.path.join(data_dir, f"planes/{channel}_clear.npy")
        clear = torch.Tensor( np.load(fname) )
        if task == 'roi':
            mask = (clear <= threshold) & (clear >= -threshold)
            clear[mask] = 0
            clear[~mask] = 1
            self.balance_ratio = np.count_nonzero(clear)/clear.numel()
        self.clear = clear

        fname = os.path.join(data_dir, f"planes/{channel}_noisy.npy")
        noisy = np.load(fname)
        medians = np.median(noisy.reshape([noisy.shape[0],-1]), axis=1)
        self.noisy = torch.Tensor( noisy - medians[:,None,None,None] )
        
    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], self.clear[index]

# TODO: is the label generation in the PlaneLoader correct according to the
# threshold considerations?
