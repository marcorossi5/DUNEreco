# This file is part of DUNEdn by M. Rossi
from torch import nn
from dunedn.denoising.model_utils import choose


class PreProcessBlock(nn.Module):
    def __init__(self, kernel_size, ic, oc, getgraph_fn, model):
        ks = kernel_size
        kso2 = kernel_size // 2
        super(PreProcessBlock, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.activ = nn.LeakyReLU(0.05)
        self.convs = nn.Sequential(
            nn.Conv2d(ic, oc, ks, padding=(kso2, kso2)),
            self.activ,
            nn.Conv2d(oc, oc, ks, padding=(kso2, kso2)),
            self.activ,
            nn.Conv2d(oc, oc, ks, padding=(kso2, kso2)),
            self.activ,
        )
        self.bn = nn.BatchNorm2d(oc)
        self.GC = choose(model, oc, oc)

    def forward(self, x):
        x = self.convs(x)
        graph = self.getgraph_fn(x)
        return self.activ(self.GC(x, graph))


class ROI(nn.Module):
    """ U-net style binary segmentation """

    def __init__(self, kernel_size, ic, hc, getgraph_fn, model):
        super(ROI, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.PreProcessBlock = PreProcessBlock(kernel_size, ic, hc, getgraph_fn, model)
        self.GCs = nn.ModuleList([choose(model, hc, hc) for i in range(8)])
        self.GC_final = choose(model, hc, 1)
        self.activ = nn.LeakyReLU(0.05)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.PreProcessBlock(x)
        for i, GC in enumerate(self.GCs):
            if i % 3 == 0:
                graph = self.getgraph_fn(x)
            x = self.activ(GC(x, graph))
        return self.act(self.GC_final(x, graph))


class HPF(nn.Module):
    """High Pass Filter"""

    def __init__(self, ic, oc, getgraph_fn, model):
        super(HPF, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, 3, padding=1), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.GCs = nn.ModuleList(
            [choose(model, ic, ic), choose(model, ic, oc), choose(model, oc, oc)]
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        x = self.conv(x)
        graph = self.getgraph_fn(x)
        for GC in self.GCs:
            x = self.act(GC(x, graph))
        return x


class LPF(nn.Module):
    """Low Pass Filter"""

    def __init__(self, ic, oc, getgraph_fn, model):
        super(LPF, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
            nn.Conv2d(ic, ic, 5, padding=2), nn.BatchNorm2d(ic), nn.LeakyReLU(0.05)
        )
        self.GCs = nn.ModuleList(
            [choose(model, ic, ic), choose(model, ic, oc), choose(model, oc, oc)]
        )
        self.BNs = nn.ModuleList(
            [nn.BatchNorm2d(ic), nn.BatchNorm2d(oc), nn.BatchNorm2d(oc)]
        )
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        y = self.conv(x)
        graph = self.getgraph_fn(y)
        for BN, GC in zip(self.BNs, self.GCs):
            y = self.act(BN(GC(y, graph)))
        return x + y


class PostProcessBlock(nn.Module):
    def __init__(self, ic, hc, getgraph_fn, model):
        super(PostProcessBlock, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.GCs = nn.ModuleList(
            [
                choose(model, hc * 3 + 1, hc * 2),
                choose(model, hc * 2, hc),
                choose(model, hc, ic),
            ]
        )
        self.BNs = nn.ModuleList(
            [nn.BatchNorm2d(hc * 2), nn.BatchNorm2d(hc), nn.Identity()]
        )
        self.acts = nn.ModuleList(
            [nn.LeakyReLU(0.05), nn.LeakyReLU(0.05), nn.Identity()]
        )

    def forward(self, x):
        for act, BN, GC in zip(self.acts, self.BNs, self.GCs):
            graph = self.getgraph_fn(x)
            x = act(BN(GC(x, graph)))
        return x
