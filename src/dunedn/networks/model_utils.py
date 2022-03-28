# This file is part of DUNEdn by M. Rossi
"""
    This module contains utility functions for networks in general.
"""
import numpy as np
import torch
import torch.nn as nn


model2batch = {
    "uscg": {"dn": 1, "roi": 1},
    "gcnn": {"dn": 128, "roi": 512},
    "cnn": {"dn": 376, "roi": 2048},
}


class MinMax(nn.Module):
    """MinMax normalization layer with scale factors Min and Max."""

    def __init__(self, Min, Max):
        """
        Parameters
        ----------
            - Min: float, scaling factor
            - Max: float, scaling factor

        Raises
        ------
            - ValueError, if Min >= Max
        """
        super(MinMax, self).__init__()
        self.Min = nn.Parameter(torch.Tensor([Min]), requires_grad=False)
        self.Max = nn.Parameter(torch.Tensor([Max]), requires_grad=False)
        if self.Max - self.Min <= 0:
            raise ValueError(
                "MinMax normalization requires different and \
                              ascending ordered scale factors"
            )

    def forward(self, x, invert=False):
        if invert:
            return x * (self.Max - self.Min) + self.Min
        return (x - self.Min) / (self.Max - self.Min)


class ZScore(nn.Module):
    """Standardization layer with scale factors mu and var."""

    def __init__(self, mu, var):
        """
        Parameters
        ----------
            - mu: float, scaling factor
            - var: float, scaling factor

        Raises
        ------
            - ValueError, if var is 0
        """
        super(ZScore, self).__init__()
        self.mu = nn.Parameter(torch.Tensor([mu]), requires_grad=False)
        self.var = nn.Parameter(torch.Tensor([var]), requires_grad=False)
        if self.var == 0:
            raise ValueError("Standardization requires non-zero variance")

    def forward(self, x, invert=False):
        if invert:
            return x * self.var + self.mu
        return (x - self.mu) / self.var


class MedianNorm(nn.Module):
    """
    Median normalization layer with scale factors Min and Max. This functions
    divides by (Max-Min)
    """

    def __init__(self, med, Min, Max):
        """
        Parameters
        ----------
            - med: float, median
            - Min: float, scaling factor
            - Max: float, scaling factor

        Raises
        ------
            - ValueError, if Min >= Max
        """
        super(MedianNorm, self).__init__()
        self.med = nn.Parameter(torch.Tensor([med]), requires_grad=False)
        self.Min = nn.Parameter(torch.Tensor([Min]), requires_grad=False)
        self.Max = nn.Parameter(torch.Tensor([Max]), requires_grad=False)
        if self.Max - self.Min <= 0:
            raise ValueError(
                "MinMax normalization requires different and \
                              ascending ordered scale factors"
            )

    def forward(self, x, invert=False):
        if invert:
            return x * (self.Max - self.Min)
        return x / (self.Max - self.Min)


def choose_norm(dataset_dir, ch, op):
    """
    Utility function to retrieve model from model name and args.

    Parameters
    ----------
        - dataset_dir: Path, path to dataset directory
        - ch: str, available options readout | collection
        - op: str, available options zscore | minmax | mednorm

    Returns
    -------
        - torch.nn.Module, the normalization instance

    Raises
    ------
        - NotImplementedError if op is not in ['zscore', 'minmax', 'mednorm']
    """
    fname = dataset_dir / f"{ch}_{op}.npy"
    params = np.load(fname)
    if op == "zscore":
        return ZScore(*params)
    elif op == "minmax":
        return MinMax(*params)
    elif op == "mednorm":
        return MedianNorm(*params)
    else:
        raise NotImplementedError("Normalization operation not implemented")


class MyDataParallel(nn.DataParallel):
    """Data Parallel wrapper that allows calling model's attributes."""

    def __getattr__(self, name):
        try:
            return super(MyDataParallel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MyDDP(nn.parallel.DistributedDataParallel):
    """Distributed Data Parallel wrapper that allows calling model's attributes."""

    def __getattr__(self, name):
        try:
            return super(MyDDP, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


# TODO: check the confusion matrix (there's one implemented in utils/utils.py)
