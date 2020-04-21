import torch
import torch.nn as nn
import torch.nn.functional as F

class NonLocalAggregation(nn.Module):
    def __init__(self, k, input_channels, out_channels, search_area=None):
        super().__init__()
        self.k = k
        self.diff_fc = nn.Linear(input_channels, out_channels)
        self.w_self = nn.Linear(input_channels, out_channels)
        self.bias = nn.Parameter(torch.randn(out_channels), requires_grad=True)
        
    def forward(self, x):
        """
        x: torch.Tensor with shape batch x features x h x w
        """
        x = x.permute(0, 2, 3, 1)
        b, h, w, f = x.shape
        x = x.view(b, h*w, f)
        
        closest_graph = get_closest_diff(x, self.k)
        agg_weights = self.diff_fc(closest_graph) # look closer
        agg_self = self.w_self(x)
                
        x_new = torch.mean(agg_weights, dim=-2) + agg_self + self.bias

        return x_new.view(b, h, w, x_new.shape[-1]).permute(0, 3, 1, 2)

def get_closest_diff(arr, k):
    """
    arr: torch.Tensor with shape batch x h * w x features
    """
    b, hw, f = arr.shape
    dists = pairwise_dist(arr, k)
    selected = batched_index_select(arr, 1, dists.view(dists.shape[0], -1)).view(b, hw, k, f)
    diff = arr.unsqueeze(2) - selected
    return diff

def calculate_pad(shape1, shape2):
    """
    x -> dim=-2
    y -> dim=-1
    """
    return_pad = [0, 0, 0, 0]
    _, im_x, im_y = shape1
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
    p_x, p_y = patch_size
    image = image.squeeze(1)
    pad = calculate_pad(image.shape, patch_size)
    image = F.pad(image, pad,
                 mode='constant', value=image.mean())

    splits = torch.stack(torch.split(image, p_y,-1),1)
    splits = torch.stack(torch.split(splits, p_x,-2),1)

    splits_shape = splits.shape

    splits = splits.view(-1, p_x, p_y).unsqueeze(1)

    return splits, splits_shape, pad

def recombine_img(splits, splits_shape, pad):
    b, a_x, a_y, p_x, p_y = splits_shape

    splits = splits.unsqueeze(1).reshape(splits_shape)
    splits = splits.permute(0,1,3,2,4)
    img = splits.reshape(-1, a_x*p_x, a_y*p_y)

    return img[:, pad[-2]:-pad[-1], pad[0]:-pad[1]]

class MyDataParallel(nn.DataParallel):
    """Allow calling model's attributes"""

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
