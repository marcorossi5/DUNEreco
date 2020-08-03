from torch import nn
import ssim

class loss:
    """Mother class interface"""
    def __init__(self, a=0.84, data_range=1., size_average=True):
        self.a = a
        self.data_range = data_range
        self.size_average = size_average

class loss_mse(loss):
    def __init__(self, a=0.84, data_range=1., size_average=True):
        super().__init__()
    def __call__(self,*args):
        return nn.MSELoss()(*args)

class loss_ssim(loss):
    def __init__(self, a=0.84, data_range=1., size_average=True):
        super().__init__(a,data_range,size_average)
    def __call__(self, *args):
        return 1-ssim.stat_ssim(*args,
                             data_range=self.data_range,
                             size_average=self.size_average)
class loss_ssim_l2(loss):
    def __init__(self, a=0.84, data_range=1., size_average=True):
        super().__init__(a,data_range,size_average)
    def __call__(self,*args):
        loss1 = 1-ssim.stat_ssim(*args,
                             data_range=self.data_range,
                             size_average=self.size_average)
        loss2 = nn.MSELoss()(*args)
        return self.a*loss1 + 1e3 * (1-self.a)*loss2

class loss_ssim_l1(loss):
    def __init__(self, a=0.84, data_range=1., size_average=True):
        super().__init__(a,data_range,size_average)
    def __call__(self, *args):
        loss1 = 1-ssim.stat_ssim(*args,
                             data_range=self.data_range,
                             size_average=self.size_average)
        loss2 = (args[0]-args[1]).abs().mean()
        return self.a*loss1 + (1-self.a)*loss2
