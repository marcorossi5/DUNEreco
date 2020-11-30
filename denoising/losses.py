""" This module implements several losses. Main option is reduction, which could
    be either 'mean' (default) or 'none'."""
import torch
from torch import nn
import ssim
from abc import ABC, abstractmethod

EPS = torch.finfo(torch.float64).eps

class loss(ABC):
    """Mother class interface"""
    def __init__(self, a=0.84, data_range=1., reduction='mean'):
        self.a = a
        self.data_range = data_range
        self.reduction = reduction
    
    @abstractmethod
    def __call__(self, *args):
        """ Compute the loss function"""


class loss_mse(loss):
    def __init__(self, a=0.84, data_range=1., reduction='mean'):
        super().__init__(reduction=reduction)
        self.loss = nn.MSELoss(reduction=self.reduction)
    def __call__(self,*args):
        return self.loss(*args)


class loss_ssim(loss):
    def __init__(self, a=0.84, data_range=1., reduction='mean'):
        super().__init__(a,data_range,reduction)
    def __call__(self, *args):
        return 1-ssim.stat_ssim(*args,
                             data_range=self.data_range,
                             reduction=self.reduction)
class loss_ssim_l2(loss):
    def __init__(self, a=0.84, data_range=1., reduction='mean'):
        super().__init__(a,data_range,reduction)
    def __call__(self,*args):
        loss1 = 1-ssim.stat_ssim(*args,
                             data_range=self.data_range,
                             reduction=self.reduction)
        loss2 = nn.MSELoss(reduction=self.reduction)(*args)
        if self.reduction == 'none':
            n = loss2.shape[0]
            loss2 = loss2.reshape([n,-1]).mean(-1)
        return self.a*loss1 + 1e-3 * (1-self.a)*loss2


class loss_ssim_l1(loss):
    def __init__(self, a=0.84, data_range=1., reduction='mean'):
        super().__init__(a,data_range,reduction)
    def __call__(self, *args):
        loss1 = 1-ssim.stat_ssim(*args,
                             data_range=self.data_range,
                             reduction=self.reduction)
        loss2 = (args[0]-args[1]).abs()
        if self.reduction == 'mean':
            loss2 = loss2.mean()
        elif self.reduction == 'none':
            n = loss2.shape[0]
            loss2 = loss2.reshape([n,-1]).mean(-1)
        return self.a*loss1 + (1-self.a)*loss2


class loss_bce(loss):
    def __init__(self, ratio, reduction='mean'):
        """
            Ratio is the number of positive against negative example in training
            set. It's used for reweighting the cross entropy
        """
        super().__init__(0,0,reduction)
        self.ratio = ratio
    def __call__(self, x, y):
        log = lambda x: torch.log(x + EPS)
        loss = - y*log(x)/(1-self.ratio) - (1-y)*log(1-x)/self.ratio
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss


class  loss_SoftDice(loss):
    def __init__(self, reduction='mean'):
        """
            Reduction: str
                'mean' | 'none'
        """
        super().__init__(0,0,reduction)
    
    def dice(self, x, y):
        """
            Parameters:
                x,y: torch.tensor
                    output and target tensors of shape (N,C,H,W)
        """
        ratio = x * y / (x*x + y*y + EPS)
        return 2 * ratio.sum(-1).sum(-1).mean(-1)

    def __call__(self, x, y):
        """
            Parameters:
                x,y: torch.tensor
                    output and target tensors of shape (N,C,H,W)
        """
        ratio = x * y / (x*x + y*y + EPS)
        loss = 1 - 2 * ratio.sum(-1).sum(-1).mean(-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction ==  'none':
            return loss


class  loss_bce_dice(loss):
    def __init__(self, reduction='mean'):
        """
            Reduction: str
                'mean' | 'none'
        """
        super().__init__(0,0,reduction)
        self.bce = loss_bce(reduction='none')
        self.dice = loss_bce_dice()
    def __call__(self, x, y):
        """
            Parameters:
                x,y: torch.tensor
                    output and target tensors of shape (N,C,H,W)
        """
        shape = [x.shape[0], -1]
        bce = self.bce(x,y).reshape(shape).mean(-1)
        dice = - torch.log(self.dice.dice(x,y) + EPS)
        loss = bce + dice
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction ==  'none':
            return loss

def get_loss(loss):
    if loss == "mse":
        return loss_mse
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
    else:
        raise NotImplementedError("Loss function not implemented")

# TODO: must check if all the reductions are consistent
