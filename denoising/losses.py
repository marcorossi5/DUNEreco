""" This module implements several losses. Main option is reduction, which could
    be either 'mean' (default) or 'none'."""
from torch import nn
import ssim
from abc import ABC, abstractmethod

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

def get_loss(loss):
    if loss == "mse":
        return loss_mse
    elif loss == "ssim":
        return loss_ssim
    elif loss == "ssim_l2":
        return loss_ssim_l2
    elif loss == "ssim_l1":
        return loss_ssim_l1
    else:
        raise NotImplementedError("Loss function not implemented")

# TODO: must check if all the reductions are consistent
