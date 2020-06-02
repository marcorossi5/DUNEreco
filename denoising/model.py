import torch
import torch.nn.functional as F
import torch.nn as nn

from model_utils import NonLocalAggregation
from model_utils import get_graph
from model_utils import split_img
from model_utils import recombine_img
from model_utils import local_mask
import ssim

def get_CNN(k, input_channels, hidden_channels,
                    patch_size=(64, 64)):

    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels, search_area=None):
            super().__init__()
            
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
            super().__init__()
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
            super().__init__()
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

    loss_mse = nn.MSELoss()

    class CNN(nn.Module):
        def __init__(self, input_channels, hidden_channels, patch_size):
            super().__init__()
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
            self.act = nn.Sigmoid()

        def fit_image(self, image):
            processed_image = torch.cat([block(image) for block in
                                        self.preprocessing_blocks], dim=1)
            residual_1 = self.residual_1(processed_image) + processed_image
            residual_2 = self.residual_2(residual_1) + residual_1
            return self.act(self.downsample(residual_2)+image)
        
        def forward(self, noised_image=None, clear_image=None):
            if self.training:
                output = self.fit_image(noised_image)
                #loss = loss_mse(output, clear_image)
                loss = 1 - ssim.ssim(output,
                                     clear_image,
                                     data_range=1.,
                                     size_average=True)
                return output, loss

            return self.fit_image(noised_image)

    cnn = CNN(input_channels, hidden_channels, patch_size)
        
    return cnn

def get_GCNN(k, input_channels, hidden_channels,
                    patch_size=(64, 64)):
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
            super().__init__()
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
            super().__init__()
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


    loss_mse = nn.MSELoss()

    class GCNN(nn.Module):
        def __init__(self, k, input_channels, hidden_channels, patch_size):
            super().__init__()
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
            
            self.act = nn.Sigmoid()

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
            return self.act(self.GC_3(result,graph) + image)


        def forward(self, noised_image=None, clear_image=None):
            out = self.fit_image(noised_image)
            #loss = loss_mse(out, clear_image)
            loss = 1 - ssim.ssim(output,
                                 clear_image,
                                 data_range=1.,
                                 size_average=True)
                
            return out, loss
            
    gcnn = GCNN(k, input_channels, hidden_channels, patch_size)

    return gcnn

def get_GCNNv2(k, input_channels, hidden_channels,
               patch_size=(64,64)):
    l_mask = local_mask(patch_size)

    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels, search_area=None):
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
            super().__init__()
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

    class HPF(nn.Module):
        """High Pass Filter"""
        def __init__(self, k, input_channels, out_channels):
            super().__init__()
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
            super().__init__()
            self.k = k
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 3, padding=1),
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

    loss_mse = nn.MSELoss()

    class GCNNv2(nn.Module):
        def __init__(self, k, input_channels, hidden_channels, patch_size):
            super().__init__()
            self.patch_size = patch_size
            self.k = k
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(k, 3, input_channels, hidden_channels),
                PreProcessBlock(k, 5, input_channels, hidden_channels),
                PreProcessBlock(k, 7, input_channels, hidden_channels),
            ])
            self.LPF_1 = LPF(k, hidden_channels*3, hidden_channels*3)
            self.LPF_2 = LPF(k, hidden_channels*3, hidden_channels*3)
            self.LPF_3 = LPF(k, hidden_channels*3, hidden_channels*3)
            self.LPF_4 = LPF(k, hidden_channels*3, hidden_channels*3)

            self.HPF = HPF(k, hidden_channels*3, hidden_channels*3)

            self.GC_1 = GraphConv(hidden_channels*3, hidden_channels*2)
            self.bn_1 = nn.BatchNorm2d(hidden_channels*2)
            self.GC_2 = GraphConv(hidden_channels*2, hidden_channels)
            self.bn_2 = nn.BatchNorm2d(hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, input_channels)

            self.relu = nn.LeakyReLU(0.05)
            self.act = nn.Sigmoid()

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

        def fit_image(self, x):
            y = torch.cat([block(x) for block in
                                        self.preprocessing_blocks], dim=1)
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
            return self.act(self.GC_3(y, graph) + x)

        def forward(self, noised_image=None, clear_image=None):
            out = self.fit_image(noised_image)
            #loss = loss_mse(out, clear_image)
            loss = 1 - ssim.ssim(output,
                                 clear_image,
                                 data_range=1.,
                                 size_average=True)
                
            return out, loss

    gcnnv2 = GCNNv2(k, input_channels, hidden_channels, patch_size)

    return gcnnv2

def get_CNNv2(k, input_channels, hidden_channels,
                    patch_size=(64, 64)):

    class GraphConv(nn.Module):
        def __init__(self, input_channels, out_channels):
            super().__init__()
            
            self.conv1 = nn.Conv2d(input_channels, out_channels, 3, padding=1)
            self.conv2 = nn.Conv2d(input_channels, out_channels, 5, padding=2)
            self.conv3 = nn.Conv2d(input_channels, out_channels, 7, padding=3)

        def forward(self, x):
            return torch.mean(torch.stack([self.conv1(x),
                                           self.conv2(x),
                                           self.conv3(x)]), dim=0)

    class PreProcessBlock(nn.Module):
        def __init__(self, kernel_size, input_channels, out_channels):
            super().__init__()
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

    class HPF(nn.Module):
        """High Pass Filter"""
        def __init__(self, input_channels, out_channels):
            super().__init__()

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
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, input_channels, 3, padding=1),
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

    loss_mse = nn.MSELoss()

    class CNNv2(nn.Module):
        def __init__(self, input_channels, hidden_channels, patch_size):
            super().__init__()
            self.patch_size = patch_size
            self.preprocessing_blocks = nn.ModuleList([
                PreProcessBlock(3, input_channels, hidden_channels),
                PreProcessBlock(5, input_channels, hidden_channels),
                PreProcessBlock(7, input_channels, hidden_channels),
            ])
            self.LPF_1 = LPF(hidden_channels*3, hidden_channels*3)
            self.LPF_2 = LPF(hidden_channels*3, hidden_channels*3)
            self.LPF_3 = LPF(hidden_channels*3, hidden_channels*3)
            self.LPF_4 = LPF(hidden_channels*3, hidden_channels*3)

            self.HPF = HPF(hidden_channels*3, hidden_channels*3)

            self.GC_1 = GraphConv(hidden_channels*3, hidden_channels*2)
            self.bn_1 = nn.BatchNorm2d(hidden_channels*2)
            self.GC_2 = GraphConv(hidden_channels*2, hidden_channels)
            self.bn_2 = nn.BatchNorm2d(hidden_channels)
            self.GC_3 = GraphConv(hidden_channels, input_channels)

            self.relu = nn.LeakyReLU(0.05)            
            self.act = nn.Sigmoid()

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

        def fit_image(self, x):
            y = torch.cat([block(x) for block in
                                        self.preprocessing_blocks], dim=1)
            y_hpf = self.HPF(y)

            y = self.LPF_1(y*(1-self.a0) + self.b0*y_hpf)
            y = self.LPF_2(y*(1-self.a1) + self.b1*y_hpf)
            y = self.LPF_3(y*(1-self.a2) + self.b2*y_hpf)
            y = self.LPF_4(y*(1-self.a3) + self.b3*y_hpf)

            y = self.relu(self.bn_1(self.GC_1(y)))
            y = self.relu(self.bn_2(self.GC_2(y)))
            return self.act(self.GC_3(y) + x)

        def forward(self, noised_image=None, clear_image=None):
            out = self.fit_image(noised_image)
            #loss = loss_mse(out, clear_image)
            loss = 1 - ssim.ssim(output,
                                 clear_image,
                                 data_range=1.,
                                 size_average=True)
                
            return out, loss

    cnnv2 = CNNv2(input_channels, hidden_channels, patch_size)

    return cnnv2
