import os
import torch

def minmax_norm(img):
    """
    MinMax normalization to be done on APAs or crops
    Parameters:
        img: torch.Tensor of shape (batch, w, h)
    Output:
    torch.Tensor of shape (batch, w, h)
    """
    m = img.min(-1)[0]
    m = m.min(-1)[0]

    M = img.max(-1)[0]
    M = M.max(-1)[0]

    mask = M!=m

    m = m.view(-1,1,1)
    M = M.view(-1,1,1)

    img[mask] = (img[mask]-m[mask])/(M[mask]-m[mask])
    return img

class CropLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, patch_size, p):
        fname = os.path.join(data_dir,
                             'clear_crops/readout_train_%d_%f'%(patch_size,
                                                                p))
        readout_clear = minmax_norm(torch.load(fname))

        fname = os.path.join(data_dir,
                             'clear_crops/collection_train_%d_%f'%(patch_size,
                                                                   p))
        collection_clear = minmax_norm(torch.load(fname))

        fname = os.path.join(data_dir,
                             'noised_crops/readout_train_%d_%f'%(patch_size,
                                                                 p))
        readout_noise = minmax_norm(torch.load(fname))

        fname = os.path.join(data_dir,
                             'noised_crops/collection_train_%d_%f'%(patch_size,
                                                                    p))
        collection_noise = minmax_norm(torch.load(fname))

        self.clear_crops = torch.cat([collection_clear, readout_clear])
        self.noised_crops = torch.cat([collection_noise, readout_noise])

        #self.clear_crops = collection_clear
        #self.noised_crops = collection_noise

        #shape: (batch, #channel,width, height)
        #with #channel==1
        self.clear_crops = self.clear_crops.unsqueeze(1)
        self.noised_crops = self.noised_crops.unsqueeze(1)
     
    def __len__(self):
        return len(self.noised_crops)
    def __getitem__(self, index):
        return self.clear_crops[index], self.noised_crops[index]

class CropValLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, patch_size, p):
        fname = os.path.join(data_dir,
                             'clear_crops/readout_val_%d_%f'%(patch_size,
                                                                p))
        readout_clear = minmax_norm(torch.load(fname))

        fname = os.path.join(data_dir,
                             'clear_crops/collection_val_%d_%f'%(patch_size,
                                                                   p))
        collection_clear = minmax_norm(torch.load(fname))

        fname = os.path.join(data_dir,
                             'noised_crops/readout_val_%d_%f'%(patch_size,
                                                                 p))
        readout_noise = minmax_norm(torch.load(fname))

        fname = os.path.join(data_dir,
                             'noised_crops/collection_val_%d_%f'%(patch_size,
                                                                    p))
        collection_noise = minmax_norm(torch.load(fname))

        self.clear_crops = torch.cat([collection_clear, readout_clear])
        self.noised_crops = torch.cat([collection_noise, readout_noise])

        #self.clear_crops = collection_clear
        #self.noised_crops = collection_noise

        #shape: (batch, #channel,width, height)
        #with #channel==1
        self.clear_crops = self.clear_crops.unsqueeze(1)
        self.noised_crops = self.noised_crops.unsqueeze(1)
     
    def __len__(self):
        return len(self.noised_crops)
    def __getitem__(self, index):
        return self.clear_crops[index], self.noised_crops[index]

class PlaneLoader(torch.utils.data.Dataset):
    def __init__(self, data_dir, file):

        fname = os.path.join(data_dir, 'clear_planes/%s'%file)
        self.clear_planes = minmax_norm(torch.load(fname)).unsqueeze(1)

        fname = os.path.join(data_dir, 'noised_planes/%s'%file)
        self.noised_planes = minmax_norm(torch.load(fname)).unsqueeze(1)
     
    def __len__(self):
        return len(self.noised_planes)
    def __getitem__(self, index):
        return self.clear_planes[index], self.noised_planes[index]

def load_planes(data_dir, file):
    fname = os.path.join(data_dir, 'clear_planes/%s'%file)
    clear_planes = torch.load(fname).unsqueeze(1)

    fname = os.path.join(data_dir, 'noised_planes/%s'%file)
    noised_planes = torch.load(fname).unsqueeze(1)

    assert len(clear_planes) == len(noised_planes)
    for i in range(len(clear_planes)):
        yield clear_planes[i:i+1], noised_planes[i:i+1]
