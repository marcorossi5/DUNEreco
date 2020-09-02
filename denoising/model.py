import torch
import torch.nn.functional as F
import torch.nn as nn

from model_utils import NonLocalAggregation
from model_utils import get_graph
from model_utils import split_img
from model_utils import recombine_img
from model_utils import local_mask
import ssim

from losses import *



def get_GCNNv2(args):
    k = args.k
    input_channels = args.in_channels
    hidden_channels = args.hidden_channels
    patch_size = args.crop_size
    loss_fn = eval(args.loss_fn)(args.a)
    #a
    l_mask = local_mask(patch_size)

    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels, search_area=None):
            super(GraphConv,self).__init__()
            self.conv1 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.NLA = NonLocalAggregation(input_channels, out_channels)

        def forward(self, x, graph):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.NLA(x, graph)]), dim=0)

    class PreProcessBlock(nn.Module):
        def __init__(self, k, kernel_size, input_channels, out_channels):
            super(PreProcessBlock,self).__init__()
            self.k = k
            self.activ = nn.LeakyReLU(0.05)
            self.convs = nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,

                nn.Conv2d(out_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,

                nn.Conv2d(out_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,
                )
            self.bn = nn.BatchNorm2d(out_channels)

            # out_channels -> out_channels
            self.GC = GraphConv(out_channels, out_channels)

        def forward(self, x):
            x = self.convs(x)

            graph = get_graph(x,self.k,l_mask)
            x = self.activ(self.GC(x, graph))
            return x

    class ROI_finder(nn.Module):
        def __init__(self, k, kernel_size, input_channels, hidden_channels):
            super(ROI_finder,self).__init__()
            self.k = k

            self.P = PreProcessBlock(k,kernel_size,
                                     input_channels, hidden_channels)
            
            self.GC_1 = GraphConv(hidden_channels, hidden_channels)
            self.GC_2 = GraphConv(hidden_channels, hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, hidden_channels)
            self.GC_4 = GraphConv(hidden_channels, hidden_channels)
            self.GC_5 = GraphConv(hidden_channels, hidden_channels)
            self.GC_6 = GraphConv(hidden_channels, hidden_channels)
            self.GC_7 = GraphConv(hidden_channels, hidden_channels)
            self.GC_8 = GraphConv(hidden_channels, hidden_channels)

            self.GC_9 = GraphConv(hidden_channels, 1)

            self.act = nn.Sigmoid()

            self.activ = nn.LeakyReLU(0.05)
            
        def forward(self, x):
            x = self.P(x)
            graph = get_graph(x,self.k,l_mask)
            x = self.activ(self.GC_1(x,graph))
            x = self.activ(self.GC_2(x,graph))
            x = self.activ(self.GC_3(x,graph))
  
            graph = get_graph(x,self.k,l_mask)
            x = self.activ(self.GC_4(x,graph))
            x = self.activ(self.GC_5(x,graph))
            x = self.activ(self.GC_6(x,graph))

            graph = get_graph(x,self.k,l_mask)
            x = self.activ(self.GC_7(x,graph))
            x = self.activ(self.GC_8(x,graph))
            return self.act(self.GC_9(x,graph))

    class HPF(nn.Module):
        """High Pass Filter"""
        def __init__(self, k, input_channels, out_channels):
            super(HPF,self).__init__()
            self.k = k

            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 3, padding=1),
                nn.BatchNorm2d(input_channels),
                nn.LeakyReLU(0.05))
            
            self.GC_1 = GraphConv(input_channels, input_channels)
            self.GC_2 = GraphConv(input_channels, out_channels)
            self.GC_3 = GraphConv(out_channels, out_channels)

            self.act = nn.LeakyReLU(0.05)
            
        def forward(self, x):
            x = self.conv(x)

            graph = get_graph(x,self.k,l_mask)
            x = self.act(self.GC_1(x, graph))
            x = self.act(self.GC_2(x, graph))
            return self.act(self.GC_3(x, graph))


    class LPF(nn.Module):
        """Low Pass Filter"""
        def __init__(self, k, input_channels, out_channels):
            super(LPF,self).__init__()
            self.k = k
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 5, padding=2),
                nn.BatchNorm2d(input_channels),
                nn.LeakyReLU(0.05))
            
            self.GC_1 = GraphConv(input_channels, input_channels)
            self.bn_1 = nn.BatchNorm2d(input_channels)
            self.GC_2 = GraphConv(input_channels, out_channels)
            self.bn_2 = nn.BatchNorm2d(out_channels)
            self.GC_3 = GraphConv(out_channels, out_channels)
            self.bn_3 = nn.BatchNorm2d(out_channels)

            self.act = nn.LeakyReLU(0.05)

        def forward(self, x):
            y = self.conv(x)

            graph = get_graph(y,self.k,l_mask)
            y = self.act(self.bn_1(self.GC_1(y, graph)))
            y = self.act(self.bn_2(self.GC_2(y, graph)))
            return x + self.act(self.bn_3(self.GC_3(y, graph)))

    class GCNNv2(nn.Module):
        """
        GNN for denoising
        ROI finder must be the first child module
        """
        def __init__(self, k, input_channels, hidden_channels, patch_size, loss_fn):
            super(GCNNv2,self).__init__()
            self.loss_fn = loss_fn
            self.patch_size = patch_size
            self.k = k

            self.hit_block = ROI_finder(k, 7, input_channels,hidden_channels) 

            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(k, 5, input_channels, hidden_channels),
                PreProcessBlock(k, 7, input_channels, hidden_channels),
                PreProcessBlock(k, 9, input_channels, hidden_channels),
            ])

            self.LPF_1 = LPF(k, hidden_channels*3+1, hidden_channels*3+1)
            self.LPF_2 = LPF(k, hidden_channels*3+1, hidden_channels*3+1)
            self.LPF_3 = LPF(k, hidden_channels*3+1, hidden_channels*3+1)
            self.LPF_4 = LPF(k, hidden_channels*3+1, hidden_channels*3+1)

            self.HPF = HPF(k, hidden_channels*3+1, hidden_channels*3+1)

            self.GC_1 = GraphConv(hidden_channels*3+1, hidden_channels*2)
            self.bn_1 = nn.BatchNorm2d(hidden_channels*2)
            self.GC_2 = GraphConv(hidden_channels*2, hidden_channels)
            self.bn_2 = nn.BatchNorm2d(hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, input_channels)

            self.relu = nn.LeakyReLU(0.05)
            #self.act = nn.Sigmoid()
            self.act = nn.Identity()

            self.a0 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a1 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a2 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a3 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a4 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.b0 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b1 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b2 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b3 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b4 = nn.Parameter(torch.Tensor([1]), requires_grad=False)

            '''
            self.a0 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.a1 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.a2 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.a3 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.a4 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.b0 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.b1 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.b2 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.b3 = nn.Parameter(torch.randn(1), requires_grad=True)
            self.b4 = nn.Parameter(torch.randn(1), requires_grad=True)
            '''
            self.xent = nn.BCELoss()

        def fit_image(self, x):
            y = torch.cat([block(x) for block in
                                        self.preprocessing_blocks], dim=1)

            hits = self.hit_block(x)
            y = torch.cat([y,hits],1)
            y_hpf = self.HPF(y)

            y = self.LPF_1(y*(1-self.a0) + self.b0*y_hpf)
            y = self.LPF_2(y*(1-self.a1) + self.b1*y_hpf)
            y = self.LPF_3(y*(1-self.a2) + self.b2*y_hpf)
            y = self.LPF_4(y*(1-self.a3) + self.b3*y_hpf)

            y = y*(1-self.a4) + self.b4*y_hpf

            graph = get_graph(y, self.k, l_mask)
            y = self.relu(self.bn_1(self.GC_1(y, graph)))

            graph = get_graph(y, self.k, l_mask)
            y = self.relu(self.bn_2(self.GC_2(y, graph)))

            graph = get_graph(y, self.k, l_mask)
            return self.act(self.GC_3(y, graph) * x), hits

        def forward(self, noised_image=None, clear_image=None, warmup=False):
            """
            Parameters:
                warmup: select the correct loss function
                        'roi': warmup loss function for roi selection only
                        'dn': warmup loss function for dn only
                        False: complete loss function with both contributions
            """
            out, hits = self.fit_image(noised_image)
            if self.training:                
                if warmup == 'roi':
                    loss_hits = self.xent(hits, clear_image[:,1:2])
                    return loss_hits, loss_hits, out.data, hits.data
                if warmup == 'dn':
                    loss = self.loss_fn(clear_image[:,:1], out)
                    return loss, loss, out.data, hits.data
                loss_hits = self.xent(hits, clear_image[:,1:2])
                loss = self.loss_fn(clear_image[:,:1], out)
                return loss + 3e-3 * loss_hits, loss_hits, out.data, hits.data
                
            return torch.cat([out, hits],1)
    gcnnv2 = GCNNv2(k, input_channels, hidden_channels, patch_size, loss_fn)

    return gcnnv2


def get_CNNv2(args):
    k = args.k
    input_channels = args.in_channels
    hidden_channels = args.hidden_channels
    patch_size = args.crop_size
    loss_fn = eval(args.loss_fn)(args.a)


    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels):
            super(GraphConv,self).__init__()
            
            self.conv1 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(input_channels, out_channels, 5, padding=2)

        def forward(self, x):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.conv2(x)]), dim=0)

    class PreProcessBlock(nn.Module):
        def __init__(self, kernel_size, input_channels, out_channels):
            super(PreProcessBlock,self).__init__()
            self.activ = nn.LeakyReLU(0.05)
            self.convs = nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,

                nn.Conv2d(out_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,

                nn.Conv2d(out_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,
                )
            self.bn = nn.BatchNorm2d(out_channels)

            # out_channels -> out_channels
            self.GC = GraphConv(out_channels, out_channels)

        def forward(self, x):
            x = self.convs(x)

            x = self.activ(self.GC(x))
            return x

    class ROI_finder(nn.Module):
        def __init__(self, kernel_size,input_channels, hidden_channels):
            super(ROI_finder,self).__init__()

            self.P = PreProcessBlock(kernel_size,
                                     input_channels, hidden_channels)
            
            self.GC_1 = GraphConv(hidden_channels, hidden_channels)
            self.GC_2 = GraphConv(hidden_channels, hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, hidden_channels)
            self.GC_4 = GraphConv(hidden_channels, hidden_channels)
            self.GC_5 = GraphConv(hidden_channels, hidden_channels)
            self.GC_6 = GraphConv(hidden_channels, hidden_channels)
            self.GC_7 = GraphConv(hidden_channels, hidden_channels)
            self.GC_8 = GraphConv(hidden_channels, hidden_channels)
            self.GC_9 = GraphConv(hidden_channels, 1)

            self.act = nn.Sigmoid()

            self.activ = nn.LeakyReLU(0.05)
            
        def forward(self, x):

            x = self.P(x)
            
            x = self.activ(self.GC_1(x))
            x = self.activ(self.GC_2(x))
            x = self.activ(self.GC_3(x))
            x = self.activ(self.GC_4(x))
            x = self.activ(self.GC_5(x))
            x = self.activ(self.GC_6(x))
            x = self.activ(self.GC_7(x))
            x = self.activ(self.GC_8(x))
            return self.act(self.GC_9(x))

            
    class HPF(nn.Module):
        """High Pass Filter"""
        def __init__(self, input_channels, out_channels):
            super(HPF,self).__init__()

            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 3, padding=1),
                nn.BatchNorm2d(input_channels),
                nn.LeakyReLU(0.05))
            
            self.GC_1 = GraphConv(input_channels, input_channels)
            self.GC_2 = GraphConv(input_channels, out_channels)
            self.GC_3 = GraphConv(out_channels, out_channels)

            self.act = nn.LeakyReLU(0.05)
            
        def forward(self, x):
            x = self.conv(x)

            x = self.act(self.GC_1(x))
            x = self.act(self.GC_2(x))
            return self.act(self.GC_3(x))


    class LPF(nn.Module):
        """Low Pass Filter"""
        def __init__(self, input_channels, out_channels):
            super(LPF,self).__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 5, padding=2),
                nn.BatchNorm2d(input_channels),
                nn.LeakyReLU(0.05))
            
            self.GC_1 = GraphConv(input_channels, input_channels)
            self.bn_1 = nn.BatchNorm2d(input_channels)
            self.GC_2 = GraphConv(input_channels, out_channels)
            self.bn_2 = nn.BatchNorm2d(out_channels)
            self.GC_3 = GraphConv(out_channels, out_channels)
            self.bn_3 = nn.BatchNorm2d(out_channels)

            self.act = nn.LeakyReLU(0.05)

        def forward(self, x):
            y = self.conv(x)

            y = self.act(self.bn_1(self.GC_1(y)))
            y = self.act(self.bn_2(self.GC_2(y)))
            return x + self.act(self.bn_3(self.GC_3(y)))

    class CNNv2(nn.Module):
        def __init__(self, input_channels, hidden_channels, patch_size, loss_fn):
            super(CNNv2,self).__init__()
            self.loss_fn = loss_fn
            self.patch_size = patch_size

            self.hit_block = ROI_finder(3,input_channels,hidden_channels)

            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(3, input_channels, hidden_channels),
                PreProcessBlock(5, input_channels, hidden_channels),
                PreProcessBlock(7, input_channels, hidden_channels)
            ])

            self.LPF_1 = LPF(hidden_channels*3+1, hidden_channels*3+1)
            self.LPF_2 = LPF(hidden_channels*3+1, hidden_channels*3+1)
            self.LPF_3 = LPF(hidden_channels*3+1, hidden_channels*3+1)
            self.LPF_4 = LPF(hidden_channels*3+1, hidden_channels*3+1)

            self.HPF = HPF(hidden_channels*3+1, hidden_channels*3+1)

            self.GC_1 = GraphConv(hidden_channels*3+1, hidden_channels*2)
            self.bn_1 = nn.BatchNorm2d(hidden_channels*2)
            self.GC_2 = GraphConv(hidden_channels*2, hidden_channels)
            self.bn_2 = nn.BatchNorm2d(hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, input_channels)

            self.relu = nn.LeakyReLU(0.05)            
            #self.act = nn.Sigmoid()
            self.act = nn.Identity()

            self.a0 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a1 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a2 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a3 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.a4 = nn.Parameter(torch.Tensor([0]), requires_grad=False)
            self.b0 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b1 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b2 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b3 = nn.Parameter(torch.Tensor([1]), requires_grad=False)
            self.b4 = nn.Parameter(torch.Tensor([1]), requires_grad=False)

            self.xent = nn.BCELoss()

        def fit_image(self, x):
            y = torch.cat([block(x) for block in
                                        self.preprocessing_blocks], dim=1)
            hits = self.hit_block(x)
            y = torch.cat([y,hits],1)
            y_hpf = self.HPF(y)

            y = self.LPF_1(y*(1-self.a0) + self.b0*y_hpf)
            y = self.LPF_2(y*(1-self.a1) + self.b1*y_hpf)
            y = self.LPF_3(y*(1-self.a2) + self.b2*y_hpf)
            y = self.LPF_4(y*(1-self.a3) + self.b3*y_hpf)

            y = self.relu(self.bn_1(self.GC_1(y)))
            y = self.relu(self.bn_2(self.GC_2(y)))
            return self.act(self.GC_3(y) * x), hits

        def forward(self, noised_image=None, clear_image=None, warmup=False):
            out, hits = self.fit_image(noised_image)
            if self.training:                
                if warmup == 'roi':
                    loss_hits = self.xent(hits, clear_image[:,1:2])
                    return loss_hits, loss_hits, out.data, hits.data
                if warmup == 'dn':
                    loss = self.loss_fn(clear_image[:,:1], out)
                    return loss, loss, out.data, hits.data
                loss_hits = self.xent(hits, clear_image[:,1:2])
                loss = self.loss_fn(clear_image[:,:1], out)
                return loss + 3e-2 * loss_hits, loss_hits, out.data, hits.data
                
            return torch.cat([out, hits],1)

    cnnv2 = CNNv2(input_channels, hidden_channels, patch_size, loss_fn)

    return cnnv2











##############################################################################










def get_CNN(args):
    k = args.k
    input_channels = args.in_channels
    hidden_channels = args.hidden_channels
    patch_size = args.crop_size
    loss_fn = eval(args.loss_fn)(args.a)
    #a

    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels, search_area=None):
            super(GraphConv,self).__init__()
            
            self.conv1 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(input_channels, out_channels, 5, padding=2)
            self.conv3 = nn.Conv2d(input_channels, out_channels, 7, padding=3)

        def forward(self, x):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.conv2(x),
                                           self.conv3(x)]), dim=0)
                                           # here is conv instead of gc

    class PreProcessBlock(nn.Module):
        def __init__(self, kernel_size, input_channels, out_channels):
            super(PreProcessBlock,self).__init__()
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                 padding=(kernel_size//2, kernel_size//2))
            self.activ = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            # out_channels -> out_channels
            self.GC = GraphConv(out_channels, out_channels)

        def forward(self, x):
            x = self.activ(self.conv(x))
            x = self.GC(x)
            x = self.activ(self.bn(x))
            return x

    class Residual(nn.Module):
        def __init__(self, input_channels, out_channels):
            super(Residual,self).__init__()
            self.pipeline = nn.Sequential(
                GraphConv(input_channels, input_channels),
                nn.BatchNorm2d(input_channels),
                nn.LeakyReLU(0.05),

                GraphConv(input_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),

                GraphConv(out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.05),
            )

        def forward(self, x):
            return self.pipeline(x)

    class CNN(nn.Module):
        def __init__(self, input_channels, hidden_channels, patch_size, loss_fn):
            super(CNN,self).__init__()
            self.loss_fn = loss_fn
            self.patch_size = patch_size
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(3, input_channels, hidden_channels),
                PreProcessBlock(5, input_channels, hidden_channels),
                PreProcessBlock(7, input_channels, hidden_channels),
                ])
            self.residual_1 = Residual(hidden_channels*3, hidden_channels*3)
            self.residual_2 = Residual(hidden_channels*3, hidden_channels*3)

            self.downsample = nn.Sequential(
                GraphConv(hidden_channels*3, hidden_channels*2),
                nn.BatchNorm2d(hidden_channels*2),
                nn.LeakyReLU(0.05),
                GraphConv(hidden_channels*2, hidden_channels),
                nn.BatchNorm2d(hidden_channels),
                nn.LeakyReLU(0.05),
                GraphConv(hidden_channels, input_channels)
            )
            #self.act = nn.Sigmoid()
            self.act = nn.Identity()

        def fit_image(self, image):
            processed_image = torch.cat([block(image) for block in
                                        self.preprocessing_blocks], dim=1)
            residual_1 = self.residual_1(processed_image) + processed_image
            residual_2 = self.residual_2(residual_1) + residual_1
            return self.act(self.downsample(residual_2)+image[:,:1])
        
        def forward(self, noised_image=None, clear_image=None):
            out = self.fit_image(noised_image)
            if self.training:
                return out, self.loss_fn(out, clear_image)
            return out

    cnn = CNN(input_channels, hidden_channels, patch_size, loss_fn)
        
    return cnn


def get_GCNN(args):
    k = args.k
    input_channels = args.in_channels
    hidden_channels = args.hidden_channels
    patch_size = args.crop_size
    loss_fn = eval(args.loss_fn)(args.a)
    #a

    l_mask = local_mask(patch_size)

    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels):
            super().__init__()
            self.conv1 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(input_channels, out_channels, 5, padding=2)
            self.NLA = NonLocalAggregation(input_channels, out_channels)

        def forward(self, x, graph):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.conv2(x),
                                           self.NLA(x, graph)]), dim=0)

    class PreProcessBlock(nn.Module):
        def __init__(self, k, kernel_size, input_channels, out_channels):
            super(PreProcessBlock,self).__init__()
            self.k = k
            self.conv = nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2))
            self.act = nn.LeakyReLU(0.05)
            self.bn = nn.BatchNorm2d(out_channels)

            # out_channels -> out_channels
            self.GC = GraphConv(out_channels, out_channels)

        def forward(self, x):
            x = self.act(self.conv(x))
            graph = get_graph(x,self.k,l_mask)
            x = self.GC(x,graph)
            x = self.act(self.bn(x))
            return x

    class Residual(nn.Module):
        def __init__(self, k, input_channels, out_channels):
            super(Residual,self).__init__()
            self.k = k
            self.act = nn.LeakyReLU(0.05)
            self.GC_1 = GraphConv(input_channels, input_channels)
            self.bn_1 = nn.BatchNorm2d(input_channels)

            self.GC_2 = GraphConv(input_channels, out_channels)
            self.bn_2 = nn.BatchNorm2d(out_channels)

            self.GC_3 = GraphConv(out_channels, out_channels)
            self.bn_3 = nn.BatchNorm2d(out_channels)

        def forward(self, x):
            graph = get_graph(x,self.k,l_mask)
            y = self.act(self.bn_1(self.GC_1(x, graph)))
            y = self.act(self.bn_2(self.GC_2(y, graph)))
            return self.act(self.bn_3(self.GC_3(y, graph)))

    class GCNN(nn.Module):
        def __init__(self, k, input_channels, hidden_channels, patch_size, loss_fn):
            super(GCNN,self).__init__()
            self.loss_fn = loss_fn
            self.patch_size = patch_size
            self.k = k
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(k, 3, input_channels, hidden_channels),
                PreProcessBlock(k, 5, input_channels, hidden_channels),
                PreProcessBlock(k, 7, input_channels, hidden_channels),
            ])
            self.residual_1 = Residual(k, hidden_channels*3, hidden_channels*3)
            self.residual_2 = Residual(k, hidden_channels*3, hidden_channels*3)

            self.GC_1 = GraphConv(hidden_channels*3, hidden_channels*2)
            self.bn_1 = nn.BatchNorm2d(hidden_channels*2)
            self.GC_2 = GraphConv(hidden_channels*2, hidden_channels)
            self.bn_2 = nn.BatchNorm2d(hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, input_channels)

            self.relu = nn.LeakyReLU(0.05)
            
            #self.act = nn.Sigmoid()
            self.act = nn.Identity()

        def fit_image(self, image):
            processed_image = torch.cat([block(image) for block in
                                        self.preprocessing_blocks], dim=1)
            residual_1 = self.residual_1(processed_image) + processed_image
            residual_2 = self.residual_2(residual_1) + residual_1
            
            graph = get_graph(residual_2,self.k,l_mask)
            result = self.bn_1(self.GC_1(residual_2,graph))
            result = self.relu(result)

            graph = get_graph(result,self.k,l_mask)
            result = self.bn_2(self.GC_2(result,graph))
            result = self.relu(result)
            
            graph = get_graph(result,self.k,l_mask)
            return self.act(self.GC_3(result,graph) + image[:,:1])


        def forward(self, noised_image=None, clear_image=None):
            out = self.fit_image(noised_image)
            if self.training:
                return out, self.loss_fn(out, clear_image)
            return out
                        
    gcnn = GCNN(k, input_channels, hidden_channels, patch_size, loss_fn)

    return gcnn


def get_ROI(args):
    k = args.k
    input_channels = args.in_channels
    hidden_channels = args.hidden_channels
    patch_size = args.crop_size
    loss_fn = args.loss_fn
    l_mask = local_mask(patch_size)

    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels):
            super(GraphConv,self).__init__()
            
            self.conv1 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.NLA = NonLocalAggregation(input_channels, out_channels)

        def forward(self, x,graph):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.NLA(x,graph)]), dim=0)

    class PreProcessBlock(nn.Module):
        def __init__(self,k, kernel_size, input_channels, out_channels):
            super(PreProcessBlock,self).__init__()
            self.k = k
            self.activ = nn.LeakyReLU(0.05)
            self.convs = nn.Sequential(
                nn.Conv2d(input_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,

                nn.Conv2d(out_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,

                nn.Conv2d(out_channels, out_channels, kernel_size,
                                  padding=(kernel_size//2, kernel_size//2)),
                self.activ,
                )
            self.bn = nn.BatchNorm2d(out_channels)

            # out_channels -> out_channels
            self.GC = GraphConv(out_channels, out_channels)

        def forward(self, x):
            x = self.convs(x)
            graph = get_graph(x, self.k, l_mask)
            x = self.activ(self.GC(x,graph))
            return x

    class ROI_finder(nn.Module):
        def __init__(self, k,kernel_size,input_channels, hidden_channels):
            super(ROI_finder,self).__init__()
            self.k = k
            self.P = PreProcessBlock(k,kernel_size,
                                     input_channels, hidden_channels)
            
            self.GC_1 = GraphConv(hidden_channels, hidden_channels)
            self.GC_2 = GraphConv(hidden_channels, hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, hidden_channels)
            self.GC_4 = GraphConv(hidden_channels, hidden_channels)
            self.GC_5 = GraphConv(hidden_channels, hidden_channels)
            self.GC_6 = GraphConv(hidden_channels, hidden_channels)
            self.GC_7 = GraphConv(hidden_channels, hidden_channels)
            self.GC_8 = GraphConv(hidden_channels, hidden_channels)

            self.GC_9 = GraphConv(hidden_channels, 1)

            self.act = nn.Sigmoid()

            self.activ = nn.LeakyReLU(0.05)
            
        def forward(self, x):
            x = self.P(x)
            graph = get_graph(x,self.k,l_mask)
            x = self.activ(self.GC_1(x,graph))
            x = self.activ(self.GC_2(x,graph))
            x = self.activ(self.GC_3(x,graph))
  
            graph = get_graph(x,self.k,l_mask)
            x = self.activ(self.GC_4(x,graph))
            x = self.activ(self.GC_5(x,graph))
            x = self.activ(self.GC_6(x,graph))

            graph = get_graph(x,self.k,l_mask)
            x = self.activ(self.GC_7(x,graph))
            x = self.activ(self.GC_8(x,graph))
            return self.act(self.GC_9(x,graph))

    class ROI(nn.Module):
        def __init__(self,k, input_channels, hidden_channels, patch_size, loss_fn):
            super(ROI,self).__init__()
            self.patch_size = patch_size            
            self.hit_block = ROI_finder(k,3,input_channels,hidden_channels)

            self.xent = nn.BCELoss()

        def fit_image(self, x):
            return self.hit_block(x)

        def forward(self, noised_image=None, clear_image=None):
            hits = self.fit_image(noised_image)
            out = torch.zeros_like(noised_image).data
            if self.training:
                loss_hits = self.xent(hits, clear_image[:,1:2])
                return loss_hits,loss_hits, out, hits.data
            return torch.cat([out, hits],1)

    roi = ROI(k,input_channels, hidden_channels, patch_size, loss_fn)

    return roi
