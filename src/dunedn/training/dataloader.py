# This file is part of DUNEdn by M. Rossi
"""
    This module contains classes inheriting from torch.utils.Dataset needed to
    run networks training and inference.
"""
import os
import torch
import numpy as np

from dunedn.networks.GCNN_Net_utils import Converter


class CropLoader(torch.utils.data.Dataset):
    """
    Loads the crops for training.
    """

    def __init__(self, dataset_dir, folder, task, channel, threshold, crop_edge, pct):
        """
        Parameters
        ----------
            - args: Args, runtime settings
            - dataset_dir: Path, path to dataset directory
            - folder: str, available options train | val | test
            - task: str, available options dn | roi
            - channel: str, available options readout | collection
            - threshold: float
            - crop_edge: int, crop edge size
            - pct: float, signal / background percentage
        """
        fname = os.path.join(
            dataset_dir, folder, "crops", f"{channel}_clear_{crop_edge}_{pct}.npy"
        )
        clear = torch.Tensor(np.load(fname))
        if task == "roi":
            mask = (clear <= threshold) & (clear >= -threshold)
            clear[mask] = 0
            clear[~mask] = 1
            self.balance_ratio = np.count_nonzero(clear) / clear.numel()

        fname = os.path.join(
            dataset_dir, folder, "crops", f"{channel}_noisy_{crop_edge}_{pct}.npy"
        )
        self.noisy = torch.Tensor(np.load(fname))

        hits = torch.clone(clear)
        hits[hits != 0] = 1

        self.clear = torch.cat([clear, hits], 1)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.clear[index], self.noisy[index]


class PlaneLoader(torch.utils.data.Dataset):
    """
    Loads the planes for training.
    Only noisy planes are normalized since clear planes don't need to be
    scaled at inference time.
    """

    def __init__(self, dataset_dir, folder, task, channel, threshold, crop_edge=None):
        """
        Parameters
        ----------
            - dataset_dir: Path, path to dataset directory
            - folder: str, available options train | val | test
            - task: str, available options dn | roi
            - channel: str, available options readout | collection
            - threshold: float
            - crop_edge: int
        """
        data_dir = os.path.join(dataset_dir, folder)
        # label = "simch" if task=='roi' else "clear"
        label = "clear"
        fname = os.path.join(data_dir, f"planes/{channel}_{label}.npy")
        clear = torch.Tensor(np.load(fname))
        if task == "roi":
            mask = (clear <= threshold) & (clear >= -threshold)
            clear[mask] = 0
            clear[~mask] = 1
            self.balance_ratio = np.count_nonzero(clear) / clear.numel()
        self.clear = clear

        fname = os.path.join(data_dir, f"planes/{channel}_noisy.npy")
        noisy = np.load(fname)
        medians = np.median(noisy.reshape([noisy.shape[0], -1]), axis=1)
        self.noisy = torch.Tensor(noisy - medians[:, None, None, None])

        if crop_edge is not None:
            self.converter = Converter((crop_edge, crop_edge))

    def to_crops(self):
        """
        Function to be called when this is used with cnn | gcnn. Converts planes
        into crops.
        """
        self.noisy = self.converter.planes2tiles(self.noisy)
        self.clear = self.converter.planes2tiles(self.clear)

    def to_planes(self):
        """
        Eventually called after to_crops function. Converts crops into planes.
        """
        self.noisy = self.converter.tiles2planes(self.noisy)
        self.clear = self.converter.tiles2planes(self.clear)

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], self.clear[index]


class InferenceLoader(torch.utils.data.Dataset):
    """
    Loads the planes for inference.
    """

    def __init__(self, noisy):
        """
        Parameters
        ----------
            - noisy: np.array, noisy planes of shape=(N,C,H,W)
        """
        medians = np.median(noisy.reshape([noisy.shape[0], -1]), axis=1)
        self.noisy = torch.Tensor(noisy - medians[:, None, None, None])

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], 0


class InferenceCropLoader(torch.utils.data.Dataset):
    """
    Loads the crops for inference.
    """

    def __init__(self, noisy):
        """
        Parameters
        ----------
            - noisy: np.array, noisy crops of shape=(N,C,H,W)
        """
        self.noisy = noisy

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, index):
        return self.noisy[index], 0
