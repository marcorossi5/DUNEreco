# This file is part of DUNEdn by M. Rossi
""" This module implements several losses. Main option is reduction, which could
    be either 'mean' (default) or 'none'."""
import torch
import numpy as np
from torch import nn
from abc import ABC, abstractmethod
from dunedn.denoising.ssim import stat_ssim
from dunedn.denoising.analysis.analysis_roi import confusion_matrix


EPS = torch.Tensor([torch.finfo(torch.float64).eps])


class loss(ABC):
    """Mother class interface"""

    def __init__(self, a=0.5, data_range=1.0, reduction="mean"):
        self.a = a
        self.data_range = data_range
        self.reduction = reduction

    @abstractmethod
    def __call__(self, *args):
        """ Compute the loss function"""


class loss_mse(loss):
    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(reduction=reduction)
        self.loss = nn.MSELoss(reduction="none")

    def __call__(self, *args):
        loss = self.loss(*args)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss.reshape([loss.shape[0], -1]).mean(-1)


class loss_imae(loss):
    """ Mean absolute error on integrated charge """

    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(reduction=reduction)
        self.loss = nn.L1Loss(reduction="none")

    def __call__(self, y_pred, y_true):
        loss = self.loss(y_pred.sum(-1), y_true.sum(-1))
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss.reshape([loss.shape[0], -1]).mean(-1)


class loss_ssim(loss):
    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(a, data_range, reduction)

    def __call__(self, *args):
        return 1 - stat_ssim(
            *args, data_range=self.data_range, reduction=self.reduction
        )


class loss_ssim_l2(loss):
    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(a, data_range, reduction)

    def __call__(self, *args):
        loss1 = 1 - stat_ssim(
            *args, data_range=self.data_range, reduction=self.reduction
        )
        loss2 = nn.MSELoss(reduction=self.reduction)(*args)
        if self.reduction == "none":
            n = loss2.shape[0]
            loss2 = loss2.reshape([n, -1]).mean(-1)
        return self.a * loss1 + (1 - self.a) * loss2


class loss_ssim_l1(loss):
    def __init__(self, a=0.84, data_range=1.0, reduction="mean"):
        super().__init__(a, data_range, reduction)

    def __call__(self, *args):
        loss1 = 1 - stat_ssim(
            *args, data_range=self.data_range, reduction=self.reduction
        )
        loss2 = (args[0] - args[1]).abs()
        if self.reduction == "mean":
            loss2 = loss2.mean()
        elif self.reduction == "none":
            n = loss2.shape[0]
            loss2 = loss2.reshape([n, -1]).mean(-1)
        return self.a * loss1 + (1 - self.a) * loss2


class loss_bce(loss):
    def __init__(self, ratio=0.5, reduction="mean"):
        """
        Ratio is the number of positive against negative example in training
        set. It's used for reweighting the cross entropy
        """
        super().__init__(0, 0, reduction)
        self.ratio = ratio

    def __call__(self, x, y):
        log = lambda x: torch.log(x + EPS.to(x.device))
        loss = -y * log(x) / self.ratio - (1 - y) * log(1 - x) / (1 - self.ratio)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "none":
            return loss


class loss_SoftDice(loss):
    def __init__(self, reduction="mean"):
        """
        Reduction: str
            'mean' | 'none'
        """
        super().__init__(0, 0, reduction)

    def dice(self, x, y):
        """
        Parameters:
            x,y: torch.tensor
                output and target tensors of shape (N,C,H,W)
        """
        eps = EPS.to(x.device)
        ix = 1 - x
        iy = 1 - y
        num1 = (x * y).sum((-1, -2)) + eps
        den1 = (x * x + y * y).sum((-1, -2)) + eps
        num2 = (ix * iy).sum((-1, -2)) + eps
        den2 = (ix * ix + iy * iy).sum((-1, -2)) + eps
        return num1 / den1 + num2 / den2

    def __call__(self, x, y):
        """
        Parameters:
            x,y: torch.tensor
                output and target tensors of shape (N,C,H,W)
        """
        ratio = self.dice(x, y)
        loss = 1 - ratio
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss


class loss_bce_dice(loss):
    def __init__(self, ratio=0.5, reduction="mean"):
        """
        Reduction: str
            'mean' | 'none'
        """
        super().__init__(0, 0, reduction)
        self.bce = loss_bce(ratio, reduction="none")
        self.dice = loss_SoftDice(reduction="none")

    def __call__(self, x, y):
        """
        Parameters:
            x,y: torch.tensor
                output and target tensors of shape (N,C,H,W)
        """
        shape = [x.shape[0], -1]
        bce = self.bce(x, y).reshape(shape).mean(-1)
        dice = -torch.log(self.dice.dice(x, y))
        loss = bce + dice
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "none":
            return loss


class loss_psnr(loss):
    def __init__(self, reduction="mean"):
        super().__init__(reduction=reduction)
        self.mse = nn.MSELoss(reduction="none")

    def __call__(self, noisy, image):
        """
        Parameters:
            image: torch.Tensor, shape (N,C,W,H)
            noisy: torch.Tensor, shape (N,C,W,H)
            reduction: str, either 'mean'| 'none'
        """
        nimages = image.shape[0]
        x1 = image.reshape(nimages, -1)
        x2 = noisy.reshape(nimages, -1)
        mse = self.mse(x1, x2).mean(-1)
        m2 = x1.max(-1).values ** 2
        zero = torch.Tensor([0.0]).to(x1.device)
        psnr = torch.where(m2 == 0, zero, 10 * torch.log10(m2 / mse))
        if self.reduction == "none":
            return psnr
        elif self.reduction == "mean":
            return psnr.mean()


class loss_cfnm(loss):
    def __init__(self, reduction="mean"):
        pass

    def __call__(self, output, target):
        # compute the confusion matrix from cuda tensors
        n = len(output)
        os = output.cpu().numpy().reshape([n, -1])
        ts = target.cpu().numpy().reshape([n, -1])
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
    if loss == "mse":
        return loss_mse
    elif loss == "imae":
        return loss_imae
    elif loss == "ssim":
        return loss_ssim
    elif loss == "ssim_l2":
        return loss_ssim_l2
    elif loss == "ssim_l1":
        return loss_ssim_l1
    elif loss == "bce":
        return loss_bce
    elif loss == "softdice":
        return loss_SoftDice
    elif loss == "bce_dice":
        return loss_bce_dice
    elif loss == "psnr":
        return loss_psnr
    elif loss == "cfnm":
        return loss_cfnm
    else:
        raise NotImplementedError("Loss function not implemented")


# TODO: must check if all the reductions are consistent
# TODO: transform psnr into loss subclass
