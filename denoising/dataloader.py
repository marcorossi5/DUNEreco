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
        hits[hits!=0] = 1
        self.clear = torch.cat([clear, hits],1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.clear[index], self.noisy[index]

class PlaneLoader(torch.utils.data.Dataset):
    def __init__(self, args, folder=None, planes=None):
        """
        This function loads the planes for inference.
        Only noisy planes are normalized since clear planes don't
        need to be scaled at inference time.
        Parameters:
            args: Args object
            folder: str, one of ['train','val','test']
            t: float, threshold to be put on labels
        """
        if folder==None and (planes is None):
            raise ValueError("Either folder or planes arguments must be given")

        self.patch_size = args.patch_size
        if folder is not None:
            data_dir = os.path.join(args.dataset_dir, folder)
            fname = os.path.join(data_dir, f"planes/{args.channel}_clear.npy")
            clear = torch.Tensor( np.load(fname) )
            hits = torch.clone(clear)
            hits[hits!= 0] = 1
            self.clear = torch.cat([clear, hits],1)
            fname = os.path.join(data_dir, f"planes/{args.channel}_noisy.npy")
            self.noisy = torch.Tensor( np.load(fname) )
        else:
            self.noisy = torch.Tensor(planes)
        
        self.converter = Converter(self.patch_size)
        self.splits = self.converter.planes2tiles(self.noisy)

    def __len__(self):
        return len(self.splits)

    def __getitem__(self, index):
        return self.splits[index]

# TODO: is the label generation in the PlaneLoader correct according to the
# threshold considerations?
