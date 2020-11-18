import torch
import torch.nn.functional as F
from torch import nn

from model_utils import choose
from model_utils import NonLocalAggregator
from model_utils import NonLocalGraph

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
        return self.activ( self.GC(x, graph) )


class ROI(nn.Module):
    def __init__(self, kernel_size, ic, hc, getgraph_fn, model):
        super(ROI, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.PreProcessBlock = PreProcessBlock(kernel_size, ic, hc,
                                               getgraph_fn, model)
        self.GCs = nn.ModuleList([choose(model, hc, hc) for i in range(8)])
        self.GC_final = choose(model, hc, 1)
        self.activ = nn.LeakyReLU(0.05)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.PreProcessBlock(x)
        for i, GC in enumerate(self.GCs):
            if i%3==0:
                graph = self.getgraph_fn(x)
            x = self.activ( GC(x, graph) )
        return self.act( self.GC_final(x, graph) )


class HPF(nn.Module):
    """High Pass Filter"""
    def __init__(self, ic, oc, getgraph_fn, model):
        super(HPF, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
                nn.Conv2d(ic, ic, 3, padding=1),
                nn.BatchNorm2d(ic),
                nn.LeakyReLU(0.05))
        self.GCs = nn.ModuleList([
                       choose(model, ic, ic),
                       choose(model, ic, oc),
                       choose(model, oc, oc)
                   ])
        self.act = nn.LeakyReLU(0.05)
            
    def forward(self, x):
        x = self.conv(x)
        graph = self.getgraph_fn(x)
        for GC in self.GCs:
            x = self.act( GC(x, graph) )
        return x


class LPF(nn.Module):
    """Low Pass Filter"""
    def __init__(self, ic, oc, getgraph_fn, model):
        super(LPF, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.conv = nn.Sequential(
                nn.Conv2d(ic, ic, 5, padding=2),
                nn.BatchNorm2d(ic),
                nn.LeakyReLU(0.05))
        self.GCs = nn.ModuleList([
                       choose(model, ic, ic),
                       choose(model, ic, oc),
                       choose(model, oc, oc)
                   ])
        self.BNs = nn.ModuleList([
                       nn.BatchNorm2d(ic),
                       nn.BatchNorm2d(oc),
                       nn.BatchNorm2d(oc)
                   ])
        self.act = nn.LeakyReLU(0.05)

    def forward(self, x):
        y = self.conv(x)
        graph = self.getgraph_fn(y)
        for BN, GC in zip(self.BNs, self.GCs):
            y = self.act( BN( GC(y, graph) ) )
        return x + y


class PostProcessBlock(nn.Module):
    def __init__(self, ic, hc, getgraph_fn, model):
        super(PostProcessBlock, self).__init__()
        self.getgraph_fn = getgraph_fn
        self.GCs = nn.ModuleList([
                       choose(model, hc*3+1, hc*2),
                       choose(model, hc*2, hc),
                       choose(model, hc, ic)
                   ])
        self.BNs = nn.ModuleList([
                       nn.BatchNorm2d(hc*2),
                       nn.BatchNorm2d(hc),
                       nn.Identity()
                   ])
        self.acts = nn.ModuleList([
                        nn.LeakyReLU(0.05),
                        nn.LeakyReLU(0.05),
                        nn.Identity()
                    ])

    def forward(self, x):
        for act, BN, GC in zip(self.acts, self.BNs, self.GCs):
            graph = self.getgraph_fn(x)
            x = act( BN( GC(x, graph) ) )
        return x


class DenoisingModel(nn.Module):
    """
    Generic neural network: based on args passed when running __init__, it
    switches between cnn|gcnn and roi|dn as well
    """
    def __init__(self, args):
        super(DenoisingModel, self).__init__()
        self.patch_size = args.patch_size
        self.model = args.model
        self.task = args.task
        ic = args.input_channels
        hc = args.hidden_channels
        self.getgraph_fn = NonLocalGraph(args.k, self.patch_size) if \
                           self.model=="gcnn" else lambda x: None
        self.ROI = ROI(7, ic, hc, self.getgraph_fn, self.model)
        self.PreProcessBlocks = nn.ModuleList([
             PreProcessBlock(5, ic, hc, self.getgraph_fn, self.model),
             PreProcessBlock(7, ic, hc, self.getgraph_fn, self.model),
             PreProcessBlock(9, ic, hc, self.getgraph_fn, self.model),
        ])
        self.LPFs = nn.ModuleList([LPF(hc*3+1, hc*3+1, self.getgraph_fn,
                                       self.model) for i in range(4)])
        self.HPF = HPF(hc*3+1, hc*3+1, self.getgraph_fn, self.model)
        self.PostProcessBlock = PostProcessBlock(ic, hc, self.getgraph_fn,
                                                 self.model)
        self.aa = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.bb = nn.Parameter(torch.Tensor([1]), requires_grad=False)
        def combine(x, y):
            return (1-self.aa)*x + self.bb*y
        self.combine = combine

    def forward(self, x, identity=False):
        if identity:
            return nn.Identity()(x)
        hits = self.ROI(x)
        if self.task == 'roi':
            return hits
        y = torch.cat([Block(x) for Block in self.PreProcessBlocks], dim=1)
        y = torch.cat([y,hits],1)
        y_hpf = self.HPF(y)
        y = self.combine(y, y_hpf)
        for LPF in self.LPFs:
            y = self.combine( LPF(y), y_hpf )
        return self.PostProcessBlock(y) + x
