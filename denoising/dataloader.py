import os
import torch
import numpy as np

from model_utils import plot_crops
from model_utils import plot_wires
from model_utils import calculate_pad
from model_utils import Converter

from ssim import _fspecial_gauss_1d, stat_gaussian_filter


def plot_example(clear):
    import matplotlib.pyplot as plt
    plt.imshow(clear[-2,0,480:], aspect="auto")
    plt.colorbar()
    plt.savefig("../collection_t.png", dpi=300, bbox_inches='tight')
    plt.close()
    exit()


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
        label = "simch" if task=='roi' else "clear"
        fname = os.path.join(data_dir, f"planes/{channel}_{label}.npy")
        clear = torch.Tensor( np.load(fname) )
        if task == 'roi':
            mask = (clear <= threshold) & (clear >= -threshold)
            clear[mask] = 0
            clear[~mask] = 1
            self.balance_ratio = np.count_nonzero(clear)/clear.numel()
        self.clear = clear
        plot_example(self.clear)

        fname = os.path.join(data_dir, f"planes/{channel}_noisy.npy")
        noisy = np.load(fname)
        medians = np.median(noisy.reshape([noisy.shape[0],-1]), axis=1)
        self.noisy = torch.Tensor( noisy - medians[:,None,None,None] )
        
    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], self.clear[index]


class InferenceLoader(torch.utils.data.Dataset):
    def __init__(self, noisy):
        medians = np.median(noisy.reshape([noisy.shape[0],-1]), axis=1)
        self.noisy = torch.Tensor( noisy - medians[:,None,None,None] )

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], 0


# TODO: is the label generation in the PlaneLoader correct according to the
# threshold considerations?
