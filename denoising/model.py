from math import ceil

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import resnext50_32x4d

from model_utils import choose_norm
from model_utils import choose
from model_utils import NonLocalAggregator
from model_utils import NonLocalGraph

from SCG_Net import SCG_Block
from SCG_Net import GCN_Layer
from SCG_Net import Pooling_Block
from SCG_Net import Recombination_Layer
from SCG_Net import weight_xavier_init


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
    """ U-net style binary segmentation """
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
        self.norm_fn = choose_norm(args.dataset_dir, args.channel,
                                   args.normalization)
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

    def forward(self, x):
        x = self.norm_fn(x)
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


class SCG_Net(nn.Module):
    def __init__(self, out_channels=1, h=960, w=6000, pretrained=True,
                 task='dn', nodes=(28,28), dropout=0.5,
                 enhance_diag=True, aux_pred=True):
        """
	Parameters:
            out_channels: int, output image channels number
            h: int, input image height
            w: int, input image width
            pretrained: bool, if True, download weight of pretrained resnet
            nodes: tuple, (height, width) of the image input of SCG block
        """
        super(SCG_Net, self).__init__()

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = resnext50_32x4d(pretrained=pretrained, progress=True)
        resnet_12 = nn.Sequential(nn.Conv2d(1,3,1),
                                  resnet.conv1,
                                  resnet.bn1,
                                  resnet.relu,
                                  resnet.maxpool,
                                  resnet.layer1,
                                  resnet.layer2)
        resnet_34 = nn.Sequential(resnet.layer3,
                                  resnet.layer4,
                                  nn.Conv2d(2048, 1024, 1))
        self.downsamples = nn.ModuleList([resnet_12,
                                          resnet_34,
                                          Pooling_Block(1024, 28, 28)])
        self.upsamples = nn.ModuleList([Pooling_Block(1, ceil(h/32), ceil(w/32)),
                                        Pooling_Block(1, ceil(h/8), ceil(w/8)),
                                        Pooling_Block(1, h, w)])
        self.GCNs = nn.Sequential(GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout),
                                  GCN_Layer(128, out_channels, bnorm=False, activation=None))
        self.scg = SCG_Block(in_ch=1024,
                             hidden_ch=out_channels,
                             node_size=nodes,
                             add_diag=enhance_diag,
                             dropout=dropout)
        # weight_xavier_init(*self.GCNs, self.scg)
        self.adapts = nn.ModuleList([nn.Conv2d(512,1,1,bias=False),
                                     nn.Conv2d(1024,1,1,bias=False),
                                     nn.Conv2d(1024,1,1,bias=False),])
        self.recombs = nn.ModuleList([Recombination_Layer() for i in range(3)])
        self.last_recomb = Recombination_Layer()
        self.act = nn.Sigmoid() if task=='roi' else nn.Identity()

    def forward(self, x):
        i = x

        # downsampling
        ys = []
        for adapt, downsample in zip(self.adapts, self.downsamples):
            x = downsample(x)
            ys.append(adapt(x))

        # Graph
        B, C, H, W = x.size()
        A, x, loss, z_hat = self.scg(x)
        x, _ = self.GCNs( (x.reshape(B, -1, C), A) )
        if self.aux_pred:
            x += z_hat
        x = x.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])

        # upsampling
        for y, recomb, upsample in zip(reversed(ys), reversed(self.recombs),
                                       self.upsamples):
            x = upsample(recomb(x,y))

        if self.training:
            return self.act(self.last_recomb(x, i)), loss
        return self.act(self.last_recomb(x, i)).cpu().data
