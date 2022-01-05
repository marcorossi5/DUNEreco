# This file is part of DUNEdn by M. Rossi
""" 
    This module implements several losses.
    Main option is reduction, which could be either 'mean' (default) or 'none'.
"""
import torch
import numpy as np
from torch import nn
from abc import ABC, abstractmethod
from dunedn.training.ssim import stat_ssim
from dunedn.utils.utils import confusion_matrix


EPS = torch.Tensor([torch.finfo(torch.float64).eps])


class Loss(ABC):
    """Abstract loss function class."""

    def __init__(self, a=0.5, data_range=1.0, reduction="mean"):
        """
        Parameters
        ----------
            - a: float, relative weight of the loss constributions
            - data_range: float
            - reduction: str: available options mean | none
        """
        self.a = a
        self.data_range = data_range
        self.reduction = reduction

    @abstractmethod
    def __call__(self, *args):
        """ Compute the loss function"""


class Loss_mse(Loss):
    """ Mean squared error loss function."""

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(reduction=reduction)
        self.loss = nn.MSELoss(reduction="none")

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
            - y_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        loss = self.loss(y_pred, y_true)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss.reshape([loss.shape[0], -1]).mean(-1)


class Loss_imae(Loss):
    """ Mean absolute error on integrated charge loss function. """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(reduction=reduction)
        self.loss = nn.L1Loss(reduction="none")

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
            - y_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        loss = self.loss(y_pred.sum(-1), y_true.sum(-1))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss.reshape([loss.shape[0], -1]).mean(-1)


class Loss_ssim(Loss):
    """ Statistical structural similarity loss function. """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(a, data_range, reduction)

    def __call__(self, x_pred, y_true):
        """
        Parameters
        ----------
            - x_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        return 1 - stat_ssim(
            x_pred, y_true, data_range=self.data_range, reduction=self.reduction
        )


class Loss_ssim_l2(Loss):
    """ Stat ssim + MSE loss function. """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(a, data_range, reduction)

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
            - y_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        loss1 = nn.MSELoss(reduction=self.reduction)(y_pred, y_true)
        if self.reduction == "none":
            n = loss1.shape[0]
            loss1 = loss1.reshape([n, -1]).mean(-1)
        loss2 = 1 - stat_ssim(
            y_pred, y_true, data_range=self.data_range, reduction=self.reduction
        )
        return self.a * loss1 + (1 - self.a) * 1e-3 * loss2


class Loss_ssim_l1(Loss):
    """ Stat ssim + mean absolute error loss function. """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(a, data_range, reduction)

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
            - y_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        loss1 = (y_pred - y_true).abs()
        if self.reduction == "mean":
            loss1 = loss1.mean()
        elif self.reduction == "none":
            n = loss1.shape[0]
            loss1 = loss1.reshape([n, -1]).mean(-1)
        loss2 = 1 - stat_ssim(
            y_pred, y_true, data_range=self.data_range, reduction=self.reduction
        )
        return self.a * loss1 + (1 - self.a) * 1e-3 * loss2


class Loss_bce(Loss):
    """ Binary cross entropy loss function. """

    def __init__(self, ratio=0.5, reduction="mean"):
        """
        Ratio is the number of positive against negative example in training
        set. It's used for reweighting the cross entropy
        """
        super().__init__(0, 0, reduction)
        self.ratio = ratio

    def __call__(self, x_pred, y_true):
        """
        Parameters
        ----------
            - x_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        log = lambda x: torch.log(x + EPS.to(x.device))
        loss = -y_true * log(x_pred) / self.ratio - (1 - y_true) * log(1 - x_pred) / (
            1 - self.ratio
        )
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss


class Loss_SoftDice(Loss):
    """ Soft dice loss function. """

    def __init__(self, reduction="mean"):
        """
        Reduction: str
            'mean' | 'none'
        """
        super().__init__(0, 0, reduction)

    def dice(self, x, y):
        """
        Parameters
        ----------
            - x: torch.Tensor, of shape=(N,C,W,H)
            - y: torch.Tensor, of shape=(N,C,W,H)
        """
        eps = EPS.to(x.device)
        ix = 1 - x
        iy = 1 - y
        num1 = (x * y).sum((-1, -2)) + eps
        den1 = (x * x + y * y).sum((-1, -2)) + eps
        num2 = (ix * iy).sum((-1, -2)) + eps
        den2 = (ix * ix + iy * iy).sum((-1, -2)) + eps
        return num1 / den1 + num2 / den2

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
            - y_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        ratio = self.dice(y_pred, y_true)
        loss = 1 - ratio
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss


class Loss_bce_dice(Loss):
    """ Binary xent + soft dice loss function. """

    def __init__(self, ratio=0.5, reduction="mean"):
        """
        Reduction: str
            'mean' | 'none'
        """
        super().__init__(0, 0, reduction)
        self.bce = Loss_bce(ratio, reduction="none")
        self.dice = Loss_SoftDice(reduction="none")

    def __call__(self, y_pred, y_true):
        """
        Parameters
        ----------
            - y_pred: torch.Tensor, of shape=(N,C,W,H)
            - y_true: torch.Tensor, of shape=(N,C,W,H)
        """
        shape = [y_pred.shape[0], -1]
        bce = self.bce(y_pred, y_true).reshape(shape).mean(-1)
        dice = -torch.log(self.dice.dice(y_pred, y_true))
        loss = bce + dice
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss


class Loss_psnr(Loss):
    """ Peak signal to noise ration function. """

    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)
        self.mse = nn.MSELoss(reduction="none")

    def __call__(self, y_noisy, y_clear):
        """
        Parameters
        ----------
            - y_clear: torch.Tensor, of shape=(N,C,W,H)
            - y_noisy: torch.Tensor, of shape=(N,C,W,H)
        """
        nimages = y_clear.shape[0]
        x1 = y_clear.reshape(nimages, -1)
        x2 = y_noisy.reshape(nimages, -1)
        mse = self.mse(x1, x2).mean(-1)
        m2 = x1.max(-1).values ** 2
        zero = torch.Tensor([0.0]).to(x1.device)
        psnr = torch.where(m2 == 0, zero, 10 * torch.log10(m2 / mse))
        if self.reduction == "none":
            return psnr
        elif self.reduction == "mean":
            return psnr.mean()


class Loss_cfnm(Loss):
    """ Confusion matrix function. """

    def __init__(self, reduction="mean"):
        pass

    def __call__(self, y_pred, y_true):
        # compute the confusion matrix from cuda tensors
        n = len(y_pred)
        os = y_pred.cpu().numpy().reshape([n, -1])
        ts = y_true.cpu().numpy().reshape([n, -1])
        cfnm = []
        for o, t in zip(os, ts):
            hit = o[t.astype(bool)]
            no_hit = o[~t.astype(bool)]
            cfnm.append(confusion_matrix(hit, no_hit, 0.5))
        cfnm = np.stack(cfnm)

        cfnm = cfnm / cfnm[0, :].sum()
        tp = [cfnm[:, 0].mean(), cfnm[:, 0].std() / np.sqrt(n)]
        fp = [cfnm[:, 1].mean(), cfnm[:, 1].std() / np.sqrt(n)]
        fn = [cfnm[:, 2].mean(), cfnm[:, 2].std() / np.sqrt(n)]
        tn = [cfnm[:, 3].mean(), cfnm[:, 3].std() / np.sqrt(n)]

        return tp, fp, fn, tn


def get_loss(loss):
    """
    Utility function to retrieve loss from loss name.

    Parameters
    ----------
        - loss: str, available options
                mse | imae | ssim | ssim_l2 | ssim_l1 | bce | softdice | cfnm

    Returns
    -------
        - Loss, the loss instance

    Raises
    ------
        - NotImplementedError if modeltype is not in
          ["mse", "imae", "ssim", "ssim_l2", "ssim_l1", "bce", "softdice", "cfnm"]
    """
    if loss == "mse":
        return Loss_mse
    elif loss == "imae":
        return Loss_imae
    elif loss == "ssim":
        return Loss_ssim
    elif loss == "ssim_l2":
        return Loss_ssim_l2
    elif loss == "ssim_l1":
        return Loss_ssim_l1
    elif loss == "bce":
        return Loss_bce
    elif loss == "softdice":
        return Loss_SoftDice
    elif loss == "bce_dice":
        return Loss_bce_dice
    elif loss == "psnr":
        return Loss_psnr
    elif loss == "cfnm":
        return Loss_cfnm
    else:
        raise NotImplementedError("Loss function not implemented")


# TODO: must check if all the reductions are consistent
# TODO: transform psnr into loss subclass
