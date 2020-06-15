from torch import nn
import ssim

def loss_mse(*args, a=None):
    return nn.MSELoss()(*args)

def loss_ssim(*args, a=None):
    return 1-ssim.stat_ssim(*args,
                            data_range=1.,
                            size_average=True)

def loss_ssim_l2(*args, a=0.84):
    loss1 = 1-ssim.stat_ssim(*args,
                             data_range=1.,
                             size_average=True)
    loss2 = nn.MSELoss()(*args)
    return a*loss1 + (1-a)*loss2

def loss_ssim_l1(*args, a=0.84):
    loss1 = 1-ssim.stat_ssim(*args,
                             data_range=1.,
                             size_average=True)
    loss2 = (args[0]-args[1]).abs().mean()
    return a*loss1 + (1-a)*loss2
