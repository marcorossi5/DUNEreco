"""
    This module contains the utility functions for CNN and GCNN networks.
"""
from typing import Tuple
import numpy as np
import torch


def normalize_fn(self, x: torch.Tensor, parameter: float):
    return x / parameter


def normalize_back_fn(self, x: torch.Tensor, parameter: float):
    return x * parameter


def pairwise_dist(arr: torch.Tensor, k: int):
    """Computes pairwise euclidean distances between pixels.

    Parameters
    ----------
    arr: torch.Tensor,
        Points, of shape=(N,H*W,C).
    k: int
        The number of nearest neighbors.

    Returns
    -------
    torch.Tensor
        Pairwise pixel distances, of shape=(N,H*W,K).
    """
    r_arr = torch.sum(arr * arr, dim=2, keepdim=True)  # (B,N,1)
    m = torch.matmul(arr, arr.permute(0, 2, 1))  # (B,N,N)
    d = -(r_arr - 2 * m + r_arr.permute(0, 2, 1))  # (B,N,N)
    return d.topk(k=k + 1, dim=-1)[1][..., 1:]  # (B,N,K)


def batched_index_select(t: torch.Tensor, dim: int, inds: torch.Tensor):
    """Selects K nearest neighbors indices for each pixel respecting batch dimension.

    Parameters
    ----------
    t: torch.Tensor
        Input tensor, of shape=(N,H*W,C).
    dim: int
        The pixels axis. Defaults to ``1``.
    inds: torch.Tensor
        The neighbors indices, of shape=(N,H*W,C).

    Returns
    -------
    torch.Tensor
        The index tensor, of shape=(N,H*W*K,C).
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    return t.gather(dim, dummy)


def calculate_pad(plane_size, crop_size):
    """
    Given plane and crop shape, compute the needed padding to obtain exact
    tiling.

    Parameters
    ----------
        - plane_size: tuple, plane shape (N,C,H,W)
        - crop_size: tuple, crop shape (edge_h, edge_w)

    Returns
    -------
        - list, plane padding: [pre h, post h, pre w, post w]
    """
    _, _, im_h, im_w = plane_size
    edge_h, edge_w = crop_size

    diff_h = im_h % edge_h
    diff_w = im_w % edge_w

    extra_h = 0 if diff_h == 0 else edge_h - (im_h % edge_h)
    extra_w = 0 if diff_w == 0 else edge_w - (im_w % edge_w)

    pad_h_up = extra_h // 2
    pad_h_down = extra_h - pad_h_up

    pad_w_left = extra_w // 2
    pad_w_right = extra_w - pad_w_left
    return ((0, 0), (0, 0), (pad_h_up, pad_h_down), (pad_w_left, pad_w_right))


class Converter:
    """Groups image to tiles converter functions"""

    def __init__(self, crop_size: Tuple[int, int]):
        """
        Parameters
        ----------
        crop_size: Tuple[int, int]
            The tile dimensions: (edge_h, edge_w).
        """
        self.crop_size = crop_size

    def image2crops(self, images: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        images: np.ndarray
            Batched events array, of shape=(N,C,W,H).

        Returns
        -------
        crops: np.ndarray
            Tiles of shape=(N',C,edge_h,edge_w).
            With ``N' = N * ceil(H/edge_h) * ceil(W/edge_w)``
        """
        edge_h, edge_w = self.crop_size
        nb_channels = images.shape[1]

        self.pad = calculate_pad(images.shape, self.crop_size)
        images = np.pad(images, self.pad)

        nb_patches_w = np.ceil(images.shape[3] / edge_w)
        splits = np.stack(np.split(images, nb_patches_w, -1), 1)

        nb_patches_h = np.ceil(images.shape[2] / edge_h)
        splits = np.stack(np.split(splits, nb_patches_h, -2), 1)

        # (N, nb_patches_h, nb_patches_w, C, edge_h, edge_w)
        self.splits_shape = splits.shape

        # final shape=(nb_patches, C, edge_h, edge_w)
        crops = splits.reshape(-1, nb_channels, edge_h, edge_w)
        return crops

    def crops2image(self, splits: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        splits: np.ndarray
            Tiles, of shape (N',C,edge_h,edge_w).

        Returns
        -------
        np.ndarray
            Planes, of shape=(N,C,H,W).
        """
        b, a_h, a_w, nb_channels, p_h, p_w = self.splits_shape

        splits = splits.reshape(self.splits_shape)
        if isinstance(splits, np.ndarray):
            splits = splits.transpose(0, 3, 1, 4, 2, 5)
        elif isinstance(splits, torch.Tensor):
            splits = splits.permute(0, 3, 1, 4, 2, 5)
        else:
            raise NotImplementedError(
                "Function input is not ``np.ndarray`` nor ``torch.Tensor`` "
                f"got {type(splits)}"
            )
        splits = splits.reshape(b, nb_channels, -1, a_w, p_w)
        img = splits.reshape(b, nb_channels, a_h * p_h, -1)

        up_bound_h = -self.pad[2][1] if self.pad[2][1] > 0 else None
        up_bound_w = -self.pad[3][1] if self.pad[3][1] > 0 else None

        output_shape = (
            slice(None),
            slice(None),
            slice(self.pad[2][0], up_bound_h),
            slice(self.pad[3][0], up_bound_w),
        )

        return img[output_shape]
