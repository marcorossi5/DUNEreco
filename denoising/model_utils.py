import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


def local_mask(crop_size):

    x, y = crop_size
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

class NonLocalAggregation(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()
        self.diff_fc = nn.Linear(input_channels, out_channels)
        self.w_self = nn.Linear(input_channels, out_channels)
        self.bias = nn.Parameter(torch.randn(out_channels), requires_grad=True)
        
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
                
        x_new = torch.mean(agg_weights, dim=-2) + agg_self + self.bias

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
    #then the matrix has to be of shape (B,N,N), where N=prod(crop_shape)
    return D.topk(k=k, dim=-1)[1] # (B,N,K)


def batched_index_select(t, dim, inds):
    """
    t: torch.Tensor with shape batch x h*w x f
    dim: 1, dimension of the pixels
    inds: torch.Tensor with shape batch x h*w*K
    """
    dummy = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), t.size(2)) # b x h*w*K x f
    out = t.gather(dim, dummy) # b x h*w*K x f
    #this gathers only the k-closest neighbours for each pixel
    return out

def get_graph(arr, k, local_mask):
    """
    arr: torch.Tensor with shape batch x h * w x features
    ------------------
    Output: torch. Tensor with shape batch x h*w x K x f
    """
    arr = arr.permute(0, 2, 3, 1)
    b, h, w, f = arr.shape
    arr = arr.view(b, h*w, f)
    hw = h*w
    dists = pairwise_dist(arr.data, k, local_mask)
    selected = batched_index_select(arr, 1, dists.view(dists.shape[0], -1)).view(b, hw, k, f)
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


def split_img(image, patch_size):
    """
    Parameters:
        image: shape (N,C,W,H)
    """
    p_x, p_y = patch_size
    N, C, _,_ = image.shape
    #image = image.squeeze(1)
    pad = calculate_pad(image.shape, patch_size)
    image = F.pad(image, pad,
                 mode='constant', value=image.mean())

    splits = torch.stack(torch.split(image, p_y,-1),1)
    splits = torch.stack(torch.split(splits, p_x,-2),1)

    splits_shape = splits.shape #(N, split_x, split_y, C, p_x, p_y)

    splits = splits.view(-1, C, p_x, p_y)#.unsqueeze(1)

    return splits, splits_shape, pad

def recombine_img(splits, splits_shape, pad):
    """
    Parameters:
        image: shape (N*split_x*split_y,C,W,H)
        split_shape: shape ((N, split_x, split_y, C, p_x, p_y))
        pad: shape (..,..,..,..)
    """
    b, a_x, a_y, C, p_x, p_y = splits_shape

    splits = splits.unsqueeze(1).reshape(splits_shape)
    splits = splits.permute(0,1,4,3,2,5)
    img = splits.reshape(-1, a_x*p_x,C, a_y*p_y)
    img = img.permute(0,2,1,3)

    return img[:,:, pad[-2]:-pad[-1], pad[0]:-pad[1]]

class MyDataParallel(nn.DataParallel):
    """Allow calling model's attributes"""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
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
        imgs: torch.Tensor of shape (#images,w,h)
        name: string, additional string to output name
        sample: torch.Tensor selected image indices to be printed
        wire: torch.Tensor, selected wires indices to be printed
    """
    samples = imgs[sample]
    
    fname = os.path.join(out_dir, "_".join([name,"crops.png"]))
    fig = plt.figure(figsize=(25,25))
    for i in range(5):
        for j in range(5):
            ax = fig.add_subplot(5,5,i*5+j+1)
            ax.imshow(samples[i*5+j])
    plt.savefig(fname)
    plt.close()
    print("Saved image at %s"%fname)
    
def plot_wires(out_dir, imgs, name, sample, wire):
    """
    Plot ADC vs time of 5x5 channels
    Parameters:
        out_dir: string, directory path of output img
        imgs: torch.Tensor of shape (#images,w,h)
        name: string, additional string to output name
        sample: torch.Tensor selected image indices to be printed
        wire: torch.Tensor, selected wires indices to be printed
    """
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