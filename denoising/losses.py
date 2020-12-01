""" This module implements several losses. Main option is reduction, which could
    be either 'mean' (default) or 'none'."""
import torch
from torch import nn
import ssim
from abc import ABC, abstractmethod

EPS = torch.Tensor( [torch.finfo(torch.float64).eps] )

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
        return self.a*loss1 + (1-self.a)*loss2


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
        log = lambda x: torch.log(x + EPS.to(x.device))
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
        ratio = x * y / (x*x + y*y + EPS.to(x.device))
        return 2 * ratio.sum(-1).sum(-1).mean(-1)

    def __call__(self, x, y):
        """
            Parameters:
                x,y: torch.tensor
                    output and target tensors of shape (N,C,H,W)
        """
        ratio = x * y / (x*x + y*y + EPS.to(x.device))
        loss = 1 - 2 * ratio.sum(-1).sum(-1).mean(-1)
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction ==  'none':
            return loss


class  loss_bce_dice(loss):
    def __init__(self, ratio, reduction='mean'):
        """
            Reduction: str
                'mean' | 'none'
        """
        super().__init__(0,0,reduction)
        self.bce = loss_bce(ratio, reduction='none')
        self.dice = loss_SoftDice(reduction='none')
    def __call__(self, x, y):
        """
            Parameters:
                x,y: torch.tensor
                    output and target tensors of shape (N,C,H,W)
        """
        shape = [x.shape[0], -1]
        bce = self.bce(x,y).reshape(shape).mean(-1)
        dice = - torch.log(self.dice.dice(x,y) + EPS.to(x.device))
        # loss = bce + dice
        loss = bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction ==  'none':
            return loss


def loss_psnr(image, noisy, reduction='mean'):
    """
    Parameters:
        image: torch.Tensor, shape (N,C,W,H)
        noisy: torch.Tensor, shape (N,C,W,H)
        reduction: str, either 'mean'| 'none'
    """
    if len(image.shape) == 3: # (C,W,H)
        mse = torch.nn.MSELoss()(image, noisy).item()
        m2 = image.max().item()**2
        return 0 if mse==0 else 10 * np.log10(m2/mse)
    else: # (N,C,H,W)
        nimages = image.shape[0]
        x1 = image.reshape(nimages, -1)
        x2 = noisy.reshape(nimages, -1)
        mse = torch.nn.MSELoss(reduction='none')(x1,x2).data.mean(-1)
        m2 = x1.max(-1).values**2
        psnr = torch.where(m2 == 0, torch.Tensor([0.]), 10*torch.log10(m2/mse))
        if reduction == 'none':
            return psnr
        elif reduction == 'mean':
            return psnr.mean()


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
    elif loss == "psnr":
        return loss_psnr
    else:
        raise NotImplementedError("Loss function not implemented")

# TODO: must check if all the reductions are consistent
# TODO: transform psnr into loss subclass
