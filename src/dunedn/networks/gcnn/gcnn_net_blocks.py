"""
    This module contains the GCNN Net building blocks.
"""
from typing import Callable
import torch
from torch import nn
from dunedn.networks.gcnn.gcnn_net_utils import (
    pairwise_dist,
    batched_index_select,
)


class PreProcessBlock(nn.Module):
    def __init__(self, kernel_size, ic, oc, getgraph_fn, conv_fn):
        super(PreProcessBlock, self).__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.getgraph_fn = getgraph_fn
        self.activ = nn.LeakyReLU(0.05)
        self.convs = nn.Sequential(
            nn.Conv2d(ic, oc, ks, padding=kso2),
            nn.BatchNorm2d(oc),
            self.activ,
            nn.Conv2d(oc, oc, ks, padding=kso2),
            nn.BatchNorm2d(oc),
            self.activ,
        )
        self.gc = conv_fn(ks, oc, oc)

    def forward(self, x):
        x = self.convs(x)
        graph = self.getgraph_fn(x)
        output = self.activ(self.gc(x, graph))
        return output


class HPF(nn.Module):
    """High Pass Filter"""

    def __init__(self, kernel_size, ic, oc, getgraph_fn, conv_fn):
        super(HPF, self).__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, ks, padding=kso2), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.gcs = nn.ModuleList(
            [
                conv_fn(ks, ic, ic),
                conv_fn(ks, ic, oc),
                conv_fn(ks, oc, oc),
            ]
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.conv(x)
        graph = self.getgraph_fn(x)
        for gc in self.gcs:
            x = self.act(gc(x, graph))
        return x


class LPF(nn.Module):
    """Low Pass Filter"""

    def __init__(self, kernel_size, ic, oc, getgraph_fn, conv_fn):
        super(LPF, self).__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, ks, padding=kso2), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.gcs = nn.ModuleList(
            [
                conv_fn(ks, ic, ic),
                conv_fn(ks, ic, oc),
                conv_fn(ks, oc, oc),
            ]
        )
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(ic), nn.BatchNorm2d(oc), nn.BatchNorm2d(oc)]
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        y = self.conv(x)
        graph = self.getgraph_fn(y)
        for bn, gc in zip(self.bns, self.gcs):
            y = self.act(bn(gc(y, graph)))
        return x + y


class PostProcessBlock(nn.Module):
    def __init__(
        self,
        kernel_size: int,
        ic: int,
        oc: int,
        getgraph_fn: Callable,
        conv_fn: Callable,
    ):
        super(PostProcessBlock, self).__init__()
        ks = kernel_size
        self.getgraph_fn = getgraph_fn
        self.conv = conv_fn(ks, ic, oc)

    def forward(self, x):
        graph = self.getgraph_fn(x)
        return self.conv(x, graph)


class NonLocalGraph:
    """Non-local graph layer."""

    def __init__(self, k: int):
        """
        Parameters
        ----------
        k: int
            The number of nearest neighbors.
        """
        self.k = k

    def __call__(self, arr: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        arr: torch.Tensor
            The input tensor, of shape=(N,C,H,W).

        Returns
        -------
        torch.Tensor
            The output tensor, of shape=(N,H*W*K,C).
        """
        arr = arr.data.permute(0, 2, 3, 1)
        b, h, w, f = arr.shape
        arr = arr.reshape(b, h * w, f)
        hw = h * w
        dists = pairwise_dist(arr, self.k)
        selected = batched_index_select(
            arr, 1, dists.reshape(dists.shape[0], -1)
        ).reshape(b, hw, self.k, f)
        diff = arr.unsqueeze(2) - selected
        return diff


# ==============================================================================
# functions and classes to be called within this module only
class GConv(nn.Module):
    """GConv layer."""

    def __init__(self, kernel_size, ic, oc):
        """
        Parameters
        ----------
            - ic: int, input channel dimension size
            - oc: int, output channel dimension size
        """
        super(GConv, self).__init__()
        ks = kernel_size
        kso2 = kernel_size // 2
        self.conv1 = nn.Conv2d(ic, oc, ks, padding=kso2)
        self.nla = NonLocalAggregator(ic, oc)

    def forward(self, x, graph):
        return torch.mean(torch.stack([self.conv1(x), self.nla(x, graph)]), dim=0)


class Conv(nn.Module):
    """GConv layer."""

    def __init__(self, kernel_size: int, ic: int, oc: int):
        """
        Parameters
        ----------
            - ic: int, input channel dimension size
            - oc: int, output channel dimension size
        """
        super(Conv, self).__init__()
        ks = kernel_size
        kso2 = kernel_size // 2

        self.conv1 = nn.Conv2d(ic, oc, ks, padding=kso2)
        self.conv2 = nn.Conv2d(ic, oc, 5, padding=2)

    def forward(self, x, graph):
        return torch.mean(torch.stack([self.conv1(x), self.conv2(x)]), dim=0)


class NonLocalAggregator(nn.Module):
    """NonLocalAggregator layer."""

    def __init__(self, input_channels, out_channels):
        """
        Parameters
        ----------
            - input_channels: int, input channel dimension size
            - output_channels: int, output channel dimension size
        """
        super(NonLocalAggregator, self).__init__()
        self.diff_fc = nn.Linear(input_channels, out_channels)
        self.w_self = nn.Linear(input_channels, out_channels)

    def forward(self, x: torch.Tensor, graph: torch.Tensor):
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor, of shape=(N,C,H,W).
        graph: torch.Tensor
            The graph tensor.

        Returns
        -------
            - torch.Tensor of shape=(N,C,H,W)
        """
        x = x.permute(0, 2, 3, 1)
        b, h, w, f = x.shape
        x = x.view(b, h * w, f)

        agg_weights = self.diff_fc(graph)
        agg_self = self.w_self(x)

        x_new = torch.mean(agg_weights, dim=-2) + agg_self

        return x_new.view(b, h, w, x_new.shape[-1]).permute(0, 3, 1, 2)
