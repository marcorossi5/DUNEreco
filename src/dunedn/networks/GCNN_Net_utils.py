# This file is part of DUNEdn by M. Rossi
"""
    This module contains the utility functions for CNN and GCNN networks.
"""
import os
import logging
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from dunedn.utils.utils import confusion_matrix


def pairwise_dist(arr, k, local_mask):
    """
    Computes pairwise euclidean distances between pixels.

    Parameters
    ----------
        - arr: torch.Tensor, of shape=(N,H*W,C)
        - k: int, nearest neighbor number
        - local_mask: torch.Tensor, of shape=(1,H*W,H*W)

    Returns
    -------
        - torch.Tensor, pairwise pixel distances of shape=(N,H*W,H*W)
    """
    dev = arr.get_device()
    dev = "cpu" if dev == -1 else dev
    local_mask = local_mask.to(dev)
    r_arr = torch.sum(arr * arr, dim=2, keepdim=True)  # (B,N,1)
    mul = torch.matmul(arr, arr.permute(0, 2, 1))  # (B,N,N)
    D = -(r_arr - 2 * mul + r_arr.permute(0, 2, 1))  # (B,N,N)
    D = D * local_mask - (1 - local_mask)
    del mul, r_arr
    # this is the euclidean distance wrt the feature vector of the current pixel
    # then the matrix has to be of shape (B,N,N), where N=prod(crop_size)
    return D.topk(k=k, dim=-1)[1]  # (B,N,K)


def batched_index_select(t, dim, inds):
    """
    Selects K nearest neighbors indices for each pixel respecting batch dimension.

    Parameters
    ----------
        - t: torch.Tensor of shape=(N,H*W,C)
        - dim: int, pixels axis. Default is 1
        - inds: torch.Tensor, neighbors indices of shape=(N,H*W,C)

    Returns
    -------
        - torch.Tensor, index tensor of shape=(N,H*W*K,C)
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    return t.gather(dim, dummy)


def local_mask(crop_size):
    """
    Computes mask to remove local pixels from the computation.

    Parameters
    ----------
        - crops_size: tuple, (H,W)

    Returns
    -------
        - torch.Tensor, local mask of shape=(1,H*W,H*W)
    """
    x, y = crop_size
    N = x * y

    local_mask = torch.ones([N, N])
    for ii in range(N):
        if ii == 0:
            local_mask[ii, (ii + 1, ii + y, ii + y + 1)] = 0  # top-left
        elif ii == N - 1:
            local_mask[ii, (ii - 1, ii - y, ii - y - 1)] = 0  # bottom-right
        elif ii == x - 1:
            local_mask[ii, (ii - 1, ii + y, ii + y - 1)] = 0  # top-right
        elif ii == N - x:
            local_mask[ii, (ii + 1, ii - y, ii - y + 1)] = 0  # bottom-left
        elif ii < x - 1 and ii > 0:
            local_mask[
                ii, (ii + 1, ii - 1, ii + y - 1, ii + y, ii + y + 1)
            ] = 0  # first row
        elif ii < N - 1 and ii > N - x:
            local_mask[
                ii, (ii + 1, ii - 1, ii - y - 1, ii - y, ii - y + 1)
            ] = 0  # last row
        elif ii % y == 0:
            local_mask[
                ii, (ii + 1, ii - y, ii + y, ii - y + 1, ii + y + 1)
            ] = 0  # first col
        elif ii % y == y - 1:
            local_mask[
                ii, (ii - 1, ii - y, ii + y, ii - y - 1, ii + y - 1)
            ] = 0  # last col
        else:
            local_mask[
                ii,
                (
                    ii + 1,
                    ii - 1,
                    ii - y,
                    ii - y + 1,
                    ii - y - 1,
                    ii + y,
                    ii + y + 1,
                    ii + y - 1,
                ),
            ] = 0
    return local_mask.unsqueeze(0)


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
    return_pad = [0, 0, 0, 0]
    _, _, im_h, im_w = plane_size
    edge_h, edge_w = crop_size

    if (edge_h - (im_h % edge_h)) % 2 == 0:
        return_pad[2] = (edge_h - (im_h % edge_h)) // 2
        return_pad[3] = (edge_h - (im_h % edge_h)) // 2
    else:
        return_pad[2] = (edge_h - (im_h % edge_h)) // 2
        return_pad[3] = (edge_h - (im_h % edge_h)) // 2 + 1

    if (edge_w - (im_w % edge_w)) % 2 == 0:
        return_pad[0] = (edge_w - (im_w % edge_w)) // 2
        return_pad[1] = (edge_w - (im_w % edge_w)) // 2
    else:
        return_pad[0] = (edge_w - (im_w % edge_w)) // 2
        return_pad[1] = (edge_w - (im_w % edge_w)) // 2 + 1
    return return_pad


class Converter:
    """Groups image to tiles converter functions"""

    def __init__(self, crop_size):
        """
        Parameters
        ----------
            - crop_size: tuple, (edge_h, edge_w)
        """
        self.crop_size = crop_size

    def planes2tiles(self, planes):
        """
        Parameters
        ----------
            - image: torch.Tensor, planes of shape=(N,C,W,H)

        Returns
        -------
            - torch.Tensor, tiles of shape=(N',C,edge_h,edge_w) with
                            N' = N * ceil(H/edge_h) * ceil(W/edge_w)
        """
        edge_h, edge_w = self.crop_size
        C = planes.shape[1]
        self.pad = calculate_pad(planes.shape, self.crop_size)
        planes = F.pad(planes, self.pad, mode="constant", value=planes.mean())

        splits = torch.stack(torch.split(planes, edge_w, -1), 1)
        splits = torch.stack(torch.split(splits, edge_h, -2), 1)

        self.splits_shape = (
            splits.shape
        )  # (N, ceil(H/edge_h), ceil(W/edge_w), C, edge_h, edge_w)

        return splits.view(-1, C, edge_h, edge_w)

    def tiles2planes(self, splits):
        """
        Parameters
        ----------
            - splits: torch.Tensor, tiles of shape (N',C,edge_h,edge_w)

        Returns
        -------
            - torch.Tensor, planes of shape=(N,C,H,W)
        """
        b, a_x, a_y, C, p_x, p_y = self.splits_shape
        C = splits.shape[1]
        splits_shape = (b, a_x, a_y, C, p_x, p_y)

        splits = splits.reshape(splits_shape)
        splits = splits.permute(0, 1, 4, 3, 2, 5)
        img = splits.reshape(-1, a_x * p_x, C, a_y * p_y)
        img = img.permute(0, 2, 1, 3)

        return img[:, :, self.pad[-2] : -self.pad[-1], self.pad[0] : -self.pad[1]]


# ==============================================================================
# deprecated functions
from dunedn.configdn import PACKAGE

# instantiate logger
logger = logging.getLogger(PACKAGE)


def plot_crops(out_dir, imgs, name, sample):
    """
    Plots ADC colormap of channel vs time of 5x5 samples.

    Parameters
    ----------
        - d: string, directory path of output img
        - imgs: torch.Tensor of shape (N,C,H,W)
        - name: string, additional string to output name
        - sample: torch.Tensor selected image indices to be printed
        - wire: torch.Tensor, selected wires indices to be printed
    """
    imgs = imgs.permute(0, 2, 3, 1).squeeze(-1)
    samples = imgs[sample]

    fname = os.path.join(out_dir, "_".join([name, "crops.png"]))
    fig, axs = plt.subplots(5, 5, figsize=(25, 25))
    for i in range(5):
        for j in range(5):
            ax = axs[i, j]
            z = ax.imshow(samples[i * 5 + j])
            fig.colorbar(z, ax=ax)
    plt.savefig(fname)
    plt.close()
    logger.info("Saved image at %s" % fname)


def plot_wires(out_dir, imgs, name, sample, wire):
    """
    Plots ADC vs time of 5x5 channels.
    Parameters
    ----------
        - out_dir: string, directory path of output img
        - imgs: torch.Tensor of shape (N,C,H,W)
        - name: string, additional string to output name
        - sample: torch.Tensor selected image indices to be printed
        - wire: torch.Tensor, selected wires indices to be printed
    """
    imgs = imgs.permute(0, 2, 3, 1).squeeze(-1)
    samples = imgs[sample]

    fname = os.path.join(out_dir, "_".join([name, "wires.png"]))
    fig = plt.figure(figsize=(25, 25))
    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(5, 5, i * 5 + j + 1)
            ax.plot(samples[i * 5 + j, wire[i * 5 + j]], linewidth=0.3)
    plt.savefig(fname)
    plt.close()
    logger.info("Saved image at %s" % fname)


def print_cm(a, f, epoch):
    """
    Print confusion matrix at a given epoch a for binary classification to file named f

    Parameters
    ----------
        - a: np.array, confusion matrix of shape=(2,2)
        - fname: str, output file name
        - epoch: int, epoch number
    """
    tot = a.sum()
    logger.info(f"Epoch: {epoch}", file=f)
    logger.info("Over a total of %d pixels:\n" % tot, file=f)
    logger.info("------------------------------------------------", file=f)
    logger.info("|{:>20}|{:>12}|{:>12}|".format("", "Hit", "No hit"), file=f)
    logger.info("------------------------------------------------", file=f)
    logger.info(
        "|{:>20}|{:>12.4e}|{:>12.4e}|".format(
            "Predicted hit", a[1, 1] / tot, a[0, 1] / tot
        ),
        file=f,
    )
    logger.info("------------------------------------------------", file=f)
    logger.info(
        "|{:>20}|{:>12.4e}|{:>12.4e}|".format(
            "Predicted no hit", a[1, 0] / tot, a[0, 0] / tot
        ),
        file=f,
    )
    logger.info("------------------------------------------------", file=f)
    logger.info(
        "{:>21}|{:>12}|{:>12}|".format("", "Sensitivity", "Specificity"), file=f
    )
    logger.info("                     ---------------------------", file=f)
    logger.info(
        "{:>21}|{:>12.4e}|{:>12.4e}|".format(
            "", a[1, 1] / (a[1, 1] + a[1, 0]), a[0, 0] / (a[0, 1] + a[0, 0])
        ),
        file=f,
    )
    logger.info("                     ---------------------------\n\n", file=f)


def save_ROI_stats(args, epoch, clear, dn, t, ana=False):
    """
    Plot stats of the ROI: confusion matrix and histogram of the classifier's
    scores.

    Parameters
    ----------
        - dn: torch.Tensor, NN output of shape=(N,C,H,W)
        - clear: torch.Tensor, targets of shape=(N,C,H,W)
        - t: float, threshold in [0,1] range
    """
    # mpl.rcParams.update(mpl.rcParamsDefault)
    y_true = clear.detach().cpu().numpy().flatten().astype(bool)
    y_pred = dn.detach().cpu().numpy().flatten()
    hit = y_pred[y_true]
    no_hit = y_pred[~y_true]
    tp, fp, fn, tn = confusion_matrix(hit, no_hit, t)
    cm = np.array([[tn, fp], [fn, tp]])
    fname = os.path.join(args.dir_testing, "cm.txt")
    with open(fname, "a+") as f:
        print_cm(cm, f, epoch)
        f.close()
    logger.info(f"Updated confusion matrix file at {fname}")


def weight_scan(module):
    """
    Computes weights' histogram and norm.

    Parameters
    ----------
        - module: torch.nn.Module

    Returns
    -------
        - float, norm
        - np.array, bins center points
        - np.array, histogram
    """
    p = []
    for i in list(module.parameters()):
        p.append(list(i.detach().cpu().numpy().flatten()))

    p = np.concatenate(p, 0)
    norm = np.sqrt((p * p).sum()) / len(p)

    hist, edges = np.histogram(p, 100)

    return norm, (edges[:-1] + edges[1:]) / 2, hist


def freeze_weights(model, task):
    """
    Freezes weights of either ROI finder or denoiser.

    Parameters
    ----------
        - model: torch.nn.Module
        - task: str, available options dn | roi
    """
    for child in model.children():
        c = "ROI" == child._get_name()
        cond = not c if task == "roi" else c
        if cond:
            for param in child.parameters():
                param.requires_grad = False
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # net = 'roi' if ROI==0 else 'dn'
    # print('Trainable parameters in %s: %d'% (net, params))
    return model
