# This file is part of DUNEdn by M. Rossi
"""
    This module contains the models classes CNN, GCNN and USCG Net, along with
    the helper functions to retrieve a model from its model name.
"""
from math import ceil
import torch
from torch import nn
from torchvision.models import resnext50_32x4d
from dunedn.networks.GCNN_Net_blocks import (
    PreProcessBlock,
    ROI,
    HPF,
    LPF,
    PostProcessBlock,
    NonLocalGraph,
)
from dunedn.networks.USCG_Net_blocks import (
    SCG_Block,
    GCN_Layer,
    Pooling_Block,
    Recombination_Layer,
)


class GCNN_Net(nn.Module):
    """
    Generic neural network: it switches between cnn|gcnn and roi|dn as well
    """

    def __init__(
        self,
        model,
        task,
        crop_edge,
        input_channels,
        hidden_channels,
        k=None,
    ):
        """
        Parameters
        ----------
            - model: str, available options cnn | gcnn
            - task: str, available options dn | roi
            - crop_edge: int, crop edge size
            - input_channels: int, inputh channel dimension size
            - hidden_channels: int, convolutions hidden filters number
            - k: int, nearest neighbor number. None if model is cnn.
        """
        super(GCNN_Net, self).__init__()
        self.crop_size = (crop_edge,) * 2
        self.model = model
        self.task = task
        ic = input_channels
        hc = hidden_channels
        self.k = k
        self.getgraph_fn = (
            NonLocalGraph(k, self.crop_size) if self.model == "gcnn" else lambda x: None
        )
        # self.norm_fn = choose_norm(dataset_dir, channel, normalization)
        self.ROI = ROI(7, ic, hc, self.getgraph_fn, self.model)
        self.PreProcessBlocks = nn.ModuleList(
            [
                PreProcessBlock(5, ic, hc, self.getgraph_fn, self.model),
                PreProcessBlock(7, ic, hc, self.getgraph_fn, self.model),
                PreProcessBlock(9, ic, hc, self.getgraph_fn, self.model),
            ]
        )
        self.LPFs = nn.ModuleList(
            [
                LPF(hc * 3 + 1, hc * 3 + 1, self.getgraph_fn, self.model)
                for i in range(4)
            ]
        )
        self.HPF = HPF(hc * 3 + 1, hc * 3 + 1, self.getgraph_fn, self.model)
        self.PostProcessBlock = PostProcessBlock(ic, hc, self.getgraph_fn, self.model)
        self.aa = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.bb = nn.Parameter(torch.Tensor([1]), requires_grad=False)

        self.combine = lambda x, y: x + y

    def forward(self, x):
        """
        Forwards pass.

        Parameters
        ----------
            - x: torch.Tensor, input tensor of shape=(N,C,H,W)

        Returns
        -------
            - torch.Tensor, output tensor of shape=(N,C,H,W)
        """
        # x = self.norm_fn(x)
        hits = self.ROI(x)
        if self.task == "roi":
            return hits
        y = torch.cat([Block(x) for Block in self.PreProcessBlocks], dim=1)
        y = torch.cat([y, hits], 1)
        y_hpf = self.HPF(y)
        y = self.combine(y, y_hpf)
        for LPF in self.LPFs:
            y = self.combine(LPF(y), y_hpf)
        return self.PostProcessBlock(y) * x


class USCG_Net(nn.Module):
    """
    U-shaped Self Constructing Graph Network: it switches between roi|dn
    """

    def __init__(
        self,
        out_channels=1,
        h=960,
        w=6000,
        pretrained=True,
        task="dn",
        nodes=(28, 28),
        dropout=0.5,
        enhance_diag=True,
        aux_pred=True,
    ):
        """
        Parameters
        ----------
            - out_channels: int, output image channels number
            - h: int, input image height
            - w: int, input image width
            - pretrained: bool, if True, download weight of pretrained resnet
            - task: str, available options dn | roi
            - nodes: tuple, (height, width) of the image input of SCG block
            - dropout: float, percentage of neurons turned off in graph layer
            - enhance_diag: bool, SCG_block flag
            - aux_pred: bool, SCG_block flag
        """
        super(USCG_Net, self).__init__()
        self.h = h
        self.w = w
        self.task = task

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = resnext50_32x4d(pretrained=pretrained, progress=True)
        resnet_12 = nn.Sequential(
            nn.Conv2d(1, 3, 1),
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
        )
        resnet_34 = nn.Sequential(
            resnet.layer3, resnet.layer4, nn.Conv2d(2048, 1024, 1)
        )
        self.downsamples = nn.ModuleList(
            [resnet_12, resnet_34, Pooling_Block(1024, 28, 28)]
        )
        self.upsamples = nn.ModuleList(
            [
                Pooling_Block(1, ceil(self.h / 32), ceil(self.w / 32)),
                Pooling_Block(1, ceil(self.h / 8), ceil(self.w / 8)),
                Pooling_Block(1, self.h, self.w),
            ]
        )
        self.GCNs = nn.Sequential(
            GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout),
            GCN_Layer(128, out_channels, bnorm=False, activation=None),
        )
        self.scg = SCG_Block(
            in_ch=1024,
            hidden_ch=out_channels,
            node_size=nodes,
            add_diag=enhance_diag,
            dropout=dropout,
        )
        # weight_xavier_init(*self.GCNs, self.scg)
        self.adapts = nn.ModuleList(
            [
                nn.Conv2d(512, 1, 1, bias=False),
                nn.Conv2d(1024, 1, 1, bias=False),
                nn.Conv2d(1024, 1, 1, bias=False),
            ]
        )
        self.recombs = nn.ModuleList([Recombination_Layer() for i in range(3)])
        self.last_recomb = Recombination_Layer()
        self.act = nn.Sigmoid() if task == "roi" else nn.Identity()

    def forward(self, x):
        """
        Forwards pass.

        Parameters
        ----------
            - x: torch.Tensor, input tensor of shape=(N,C,H,W)

        Returns
        -------
            - torch.Tensor, output tensor of shape=(N,C,H,W)
        """
        if self.task == "roi":
            x /= 3197 + 524  # normalizing according to dataset
        i = x

        # downsampling
        ys = []
        for adapt, downsample in zip(self.adapts, self.downsamples):
            x = downsample(x)
            ys.append(adapt(x))

        # Graph
        B, C, _, _ = x.size()
        A, x, loss, z_hat = self.scg(x)
        x, _ = self.GCNs((x.reshape(B, -1, C), A))
        if self.aux_pred:
            x += z_hat
        x = x.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])

        # upsampling
        for y, recomb, upsample in zip(
            reversed(ys), reversed(self.recombs), self.upsamples
        ):
            x = upsample(recomb(x, y))

        if self.training:
            return self.act(x * i), loss
        return self.act(x * i)


# TODO: check that for training, the median normalization is done outside the
# forward pass
