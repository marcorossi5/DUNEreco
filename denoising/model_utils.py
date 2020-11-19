import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.metrics import confusion_matrix


class MinMax(nn.Module):
    def __init__(self, Min, Max):
        """
        MinMax normalization layer with scale factors Min and Max
        Parameters:
            Min, Max: float, scaling factors
        """
        self.Min = nn.Parameter(torch.Tensor(Min), requires_grad=False)
        self.Max = nn.Parameter(torch.Tensor(Max), requires_grad=False)
        if  self.Max-self.Min <= 0:
            raise ValueError("MinMax normalization requires different and \
                              ascending ordered scale factors")

    def forward(self, x, invert=True):
        if invert:
            x*(self.Max-self.Min) + self.Min
        return (x-self.Min)/(self.Max-self.Min)


class Standardization(nn.Module):
    def __init__(self, mu, var):
        """
        Standardization layer with scale factors mu and var
        Parameters:
            mu, var: float, scaling factors
        """
        self.mu = nn.Parameter(torch.Tensor(mu), requires_grad=False)
        self.var = nn.Parameter(torch.Tensor(var), requires_grad=False)
        if self.var==0:
            raise ValueError("Standardization requires non-zero variance")

    def forward(self, x, invert=True):
        if invert:
            return x*self.var + self.mu
        return (x-self.mu)/self.var


def choose_norm(dataset_dir, op):
    fname = os.path.join(dataset_dir, f"{op}.npy")
    params = np.load(fname)
    if op == "standardization":
        return Standardization(*params)
    elif op == "minmax":
        return MinMax(*params)
    else:
        raise NotImplementedError("Normalization operation not implemented")


class GConv(nn.Module):
    def __init__(self, ic, oc):
        super(GConv, self).__init__()
        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.NLA = NonLocalAggregator(ic, oc)

    def forward(self, x, graph):
        return torch.mean(torch.stack([self.conv1(x),
                                       self.NLA(x, graph)]), dim=0)


class Conv(nn.Module):
    def __init__(self, ic, oc):
        super(Conv, self).__init__()
            
        self.conv1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.conv2 = nn.Conv2d(ic, oc, 5, padding=2)

    def forward(self, x, graph):
        return torch.mean(torch.stack([self.conv1(x),
                                       self.conv2(x)]), dim=0)


def choose(model, ic, oc):
    if model=="gcnn":
        return GConv(ic, oc)
    elif model=="cnn":
        return Conv(ic, oc) 
    else:
        raise NotImplementedError("Operation not implemented")


class NonLocalAggregator(nn.Module):
    def __init__(self, input_channels, out_channels):
        super(NonLocalAggregator,self).__init__()
        self.diff_fc = nn.Linear(input_channels, out_channels)
        self.w_self = nn.Linear(input_channels, out_channels)
        #self.bias = nn.Parameter(torch.randn(out_channels), requires_grad=True)
        
    def forward(self, x, graph):
        """
        x: torch.Tensor with shape batch x features x h x w
        ------------------
        Output: torch. Tensor with shape batch x out_channels x h x w
        """
        x = x.permute(0, 2, 3, 1)
        b, h, w, f = x.shape
        x = x.view(b, h*w, f)
        
        #closest_graph = get_graph(x, self.k, local_mask) #this builds the graph
        agg_weights = self.diff_fc(graph) # look closer
        agg_self = self.w_self(x)
                
        x_new = torch.mean(agg_weights, dim=-2) + agg_self# + self.bias

        return x_new.view(b, h, w, x_new.shape[-1]).permute(0, 3, 1, 2)


def pairwise_dist(arr, k, local_mask):
    """
    arr: torch.Tensor with shape batch x h*w x features
    """
    dev = arr.get_device()
    local_mask = local_mask.to(dev)
    r_arr = torch.sum(arr * arr, dim=2, keepdim=True) # (B,N,1)
    mul = torch.matmul(arr, arr.permute(0,2,1))         # (B,N,N)
    D = - (r_arr - 2 * mul + r_arr.permute(0,2,1))       # (B,N,N)
    D = D*local_mask - (1-local_mask)
    del mul, r_arr
    #this is the euclidean distance wrt the feature vector of the current pixel
    #then the matrix has to be of shape (B,N,N), where N=prod(patch_size)
    return D.topk(k=k, dim=-1)[1] # (B,N,K)


def batched_index_select(t, dim, inds):
    """
    t: torch.Tensor with shape batch x h*w x f
    dim: 1, dimension of the pixels
    inds: torch.Tensor with shape batch x h*w*K
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2))
    # Now dummy shape is b x h*w*K x f
    out = t.gather(dim, dummy) # b x h*w*K x f
    #this gathers only the k-closest neighbours for each pixel
    return out


def local_mask(patch_size):
    x, y = patch_size
    N = x*y

    local_mask = torch.ones([N, N])
    for ii in range(N):
        if ii==0:
            local_mask[ii, (ii+1,ii+y, ii+y+1)] = 0 # top-left
        elif ii==N-1:
            local_mask[ii, (ii-1, ii-y, ii-y-1)] = 0 # bottom-right
        elif ii==x-1:
            local_mask[ii, (ii-1, ii+y, ii+y-1)] = 0 # top-right
        elif ii==N-x:
            local_mask[ii, (ii+1, ii-y, ii-y+1)] = 0 # bottom-left
        elif ii<x-1 and ii>0:
            local_mask[ii, (ii+1, ii-1, ii+y-1, ii+y, ii+y+1)] = 0 # first row
        elif ii<N-1 and ii>N-x:
            local_mask[ii, (ii+1, ii-1, ii-y-1, ii-y, ii-y+1)] = 0 # last row
        elif ii%y==0:
            local_mask[ii, (ii+1, ii-y, ii+y, ii-y+1, ii+y+1)] = 0 # first col
        elif ii%y==y-1:
            local_mask[ii, (ii-1, ii-y, ii+y, ii-y-1, ii+y-1)] = 0 # last col
        else:
            local_mask[ii, (ii+1, ii-1, ii-y, ii-y+1, ii-y-1, ii+y, ii+y+1, ii+y-1)] = 0
    return local_mask.unsqueeze(0)


class NonLocalGraph:
    def __init__(self, k, patch_size):
        self.k = k
        self.local_mask = local_mask(patch_size)
    def __call__(self, arr):
        arr = arr.data.permute(0, 2, 3, 1)
        b, h, w, f = arr.shape
        arr = arr.view(b, h*w, f)
        hw = h*w
        dists = pairwise_dist(arr, self.k, self.local_mask)
        selected = batched_index_select(
                       arr, 1, dists.view(dists.shape[0], -1)
                                       ).view(b, hw, self.k, f)
        diff = arr.unsqueeze(2) - selected
        return diff


def calculate_pad(shape1, shape2):
    """
    x -> dim=-2
    y -> dim=-1
    """
    return_pad = [0, 0, 0, 0]
    _, _, im_x, im_y = shape1
    pad_x, pad_y = shape2
    
    if (pad_x - (im_x%pad_x))%2 == 0:
        return_pad[2] = (pad_x - (im_x%pad_x))//2
        return_pad[3] = (pad_x - (im_x%pad_x))//2
    else:
        return_pad[2] = (pad_x - (im_x%pad_x))//2
        return_pad[3] = (pad_x - (im_x%pad_x))//2 + 1

    if (pad_y - (im_y%pad_y))%2 == 0:
        return_pad[0] = (pad_y - (im_y%pad_y))//2
        return_pad[1] = (pad_y - (im_y%pad_y))//2
    else:
        return_pad[0] = (pad_y - (im_y%pad_y))//2
        return_pad[1] = (pad_y - (im_y%pad_y))//2 + 1
    return return_pad


class Converter:
    """ Groups image to tiles converter functions """
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def planes2tiles(self, image):
        """
        Parameters:
            image: shape (N,C,W,H)
        """
        p_x, p_y = self.patch_size
        N, C, _,_ = image.shape
        self.pad = calculate_pad(image.shape, self.patch_size)
        image = F.pad(image, self.pad,
                 mode='constant', value=image.mean())

        splits = torch.stack(torch.split(image, p_y,-1),1)
        splits = torch.stack(torch.split(splits, p_x,-2),1)

        self.splits_shape = splits.shape #(N, split_x, split_y, C, p_x, p_y)

        return splits.view(-1, C, p_x, p_y)

    def tiles2planes(self, splits):
        """
        Parameters:
            splits: image of shape (N*split_x*split_y,C,W,H)
            split_shape: shape ((N, split_x, split_y, C, p_x, p_y))
            pad: shape (..,..,..,..)
        """
        b, a_x, a_y, C, p_x, p_y = self.splits_shape
        C = splits.shape[1]
        splits_shape = (b, a_x, a_y, C, p_x, p_y)

        splits = splits.reshape(splits_shape)
        splits = splits.permute(0,1,4,3,2,5)
        img = splits.reshape(-1, a_x*p_x,C, a_y*p_y)
        img = img.permute(0,2,1,3)

        return img[:,:, self.pad[-2]:-self.pad[-1], self.pad[0]:-self.pad[1]]

class MyDataParallel(nn.DataParallel):
    """Allow calling model's attributes"""
    def __getattr__(self, name):
        try:
            return super(MyDataParallel,self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class MyDDP(nn.parallel.DistributedDataParallel):
    """Allow calling model's attributes"""
    def __getattr__(self, name):
        try:
            return super(MyDDP,self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)



def print_summary_file(args):
    d = args.__dict__
    fname = os.path.join(args.dir_output, 'readme.txt')
    with open(fname, 'w') as f:
        f.writelines('Model summary file:\n')
        for k in d.keys():
            f.writelines('\n%s     %s'%(str(k), str(d[k])))
        f.close()

def plot_crops(out_dir, imgs, name, sample):
    """
    Plot ADC colormap of channel vs time of 5x5 samples
    Parameters:
        d: string, directory path of output img
        imgs: torch.Tensor of shape (#images,C,w,h)
        name: string, additional string to output name
        sample: torch.Tensor selected image indices to be printed
        wire: torch.Tensor, selected wires indices to be printed
    """
    imgs = imgs.permute(0,2,3,1).squeeze(-1)
    samples = imgs[sample]
    
    fname = os.path.join(out_dir, "_".join([name,"crops.png"]))
    fig, axs = plt.subplots(5,5,figsize=(25,25))
    for i in range(5):
        for j in range(5):
            ax = axs[i,j]
            z = ax.imshow(samples[i*5+j])
            fig.colorbar(z, ax=ax)
    plt.savefig(fname)
    plt.close()
    print("Saved image at %s"%fname)
    
def plot_wires(out_dir, imgs, name, sample, wire):
    """
    Plot ADC vs time of 5x5 channels
    Parameters:
        out_dir: string, directory path of output img
        imgs: torch.Tensor of shape (#images,C,w,h)
        name: string, additional string to output name
        sample: torch.Tensor selected image indices to be printed
        wire: torch.Tensor, selected wires indices to be printed
    """
    imgs = imgs.permute(0,2,3,1).squeeze(-1)
    samples = imgs[sample]
    
    fname = os.path.join(out_dir, "_".join([name,"wires.png"]))
    fig = plt.figure(figsize=(25,25))
    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(5,5,i*5+j+1)
            ax.plot(samples[i*5+j,wire[i*5+j]], linewidth=0.3)
    plt.savefig(fname)
    plt.close()
    print("Saved image at %s"%fname)

def print_cm(a, f, epoch):
    """
    Print confusion matrix a for binary classification
    to file named f
    Parameters:
        a: numpy array, shape (2,2)
        fname: str
        epoch: int
    """
    tot = a.sum()
    print(f'Epoch: {epoch}', file=f)
    print("Over a total of %d pixels:\n"%tot, file=f)
    print("------------------------------------------------", file=f)
    print("|{:>20}|{:>12}|{:>12}|".format("","Hit", "No hit"), file=f)
    print("------------------------------------------------", file=f)
    print("|{:>20}|{:>12.4e}|{:>12.4e}|".format("Predicted hit",
                                                a[1,1]/tot,a[0,1]/tot),
          file=f)
    print("------------------------------------------------", file=f)
    print("|{:>20}|{:>12.4e}|{:>12.4e}|".format("Predicted no hit",
                                                a[1,0]/tot,a[0,0]/tot),
          file=f)
    print("------------------------------------------------", file=f)
    print("{:>21}|{:>12}|{:>12}|".format("", "Sensitivity","Specificity"),
          file=f)
    print("                     ---------------------------", file=f)
    print("{:>21}|{:>12.4e}|{:>12.4e}|".format("",
                                               a[1,1]/(a[1,1]+a[1,0]),
                                               a[0,0]/(a[0,1]+a[0,0])), file=f)
    print("                     ---------------------------\n\n", file=f)


def save_ROI_stats(args,epoch,clear,dn,t,ana=False):
    """
    Plot stats of the ROI:
    Confusion matrix and histogram of the classifier's scores
    Parameters:
        dn: NN output, torch.Tensor of shape (N,C,w,h)
        clear: targets, torch.Tensor of shape (N,C,w,h) 
        t: threshold, float in [0,1]      
    """
    #mpl.rcParams.update(mpl.rcParamsDefault)
    y_true = clear.detach().cpu().numpy().flatten().astype(int)
    y_pred = dn.detach().cpu().numpy().flatten()
    cm = confusion_matrix(y_true, y_pred>t)
    fname = os.path.join(args.dir_testing, 'cm.txt')
    with open(fname, 'a+') as f:
        print_cm(cm, f, epoch)
        f.close()
    print(f'Updated confusion matrix file at {fname}')

def weight_scan(module):
    """
    Compute wheights' histogram and norm
    Parameters:
        module: torch.nn.Module
    Return:
        norm: float
        edges: np.array, bins' center points
        hist: np.array
    """
    p = []
    for i in list(module.parameters()):
        p.append(list(i.detach().cpu().numpy().flatten()))
    
    p = np.concatenate(p,0)
    norm = np.sqrt((p*p).sum())/len(p)

    hist, edges = np.histogram(p,100)
    
    return norm, (edges[:-1]+edges[1:])/2, hist

def freeze_weights(model, task):
    """
    Freezes weights of either ROI finder or denoiser
    Parameters:
        model: torch.nn.Module
        task: str, mode of the training ('roi'/'dn')
    """
    for child in model.children():
        c = "ROI" == child._get_name()
        cond = not c if task=="roi" else c
        if cond:
            for param in child.parameters():
                param.requires_grad = False
    # model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    # params = sum([np.prod(p.size()) for p in model_parameters])
    # net = 'roi' if ROI==0 else 'dn'
    # print('Trainable parameters in %s: %d'% (net, params))
    return model


