import numpy as np
import torch
import glob
event_step = 15360
collection_step = 960
readout_step = 800
time_len = 6000
ada_step = event_step // (readout_step*2 + collection_step)


def sample_binomial(num_trials, probs):
    return np.random.binomial(num_trials, probs)

def normalize(img):
    if img.max() == 0:
        return img
    return (img-img.min())/(img.max()-img.min())

def load_files(path_clear, path_noise):
    clear_files = glob.glob(path_clear)
    noised_files = glob.glob(path_noise)

    for f_clear, f_noise in zip(clear_files, noised_files):
        clear_data = np.load(f_clear)[:, 2:]
        noised_data = np.load(f_noise)[:, 3:]
        yield clear_data, noised_data

def plane_idx():
    signal_planes = [i for i in range(ada_step)]

    readout = []
    collection = []
    cpp = 2*readout_step+collection_step #channel per plane

    for i in range(ada_step):
        readout.extend(range(cpp*i, cpp*i + 2*readout_step))
        collection.extend(range(cpp*i + 2*readout_step, cpp*i + 2*readout_step + collection_step))

    return readout, collection

def normalize_planes(clear_file, noised_file, r_idx, c_idx):
    r_clear = clear_file[r_idx]
    r_noised = noised_file[r_idx]

    c_clear = clear_file[c_idx]
    c_noised = noised_file[c_idx]

    r_n_clear = []
    r_n_noised = []

    c_n_clear = []
    c_n_noised = []

    for i in range(int(r_clear.shape[0]/readout_step)):
        if r_clear[i*readout_step:(i+1)*readout_step].max() == 0:
            print('skipped')
            continue
        r_n_clear.append(normalize(r_clear[i*readout_step:(i+1)*readout_step]))
        r_n_noised.append(normalize(r_noised[i*readout_step:(i+1)*readout_step]))

    for i in range(int(c_clear.shape[0]/collection_step)):
        if c_clear[i*collection_step:(i+1)*collection_step].max() == 0:
            print('skipped')
            continue
        c_n_clear.append(normalize(c_clear[i*collection_step:(i+1)*collection_step]))
        c_n_noised.append(normalize(c_noised[i*collection_step:(i+1)*collection_step]))

    return np.stack(r_n_clear), np.stack(r_n_noised), np.stack(c_n_clear), np.stack(c_n_noised)

def get_planes(clear_file, noised_file):
    """
        Returning the three separate nonempty planes
    """
    r_idx, c_idx = plane_idx()

    return normalize_planes(clear_file, noised_file, r_idx, c_idx)

def get_crop(clear_plane, noised_plane, dir_name, fname, total_crops=1000, crop_shape=(32,32), num_trials=5, device=torch.device('cpu')):
    clear_plane = torch.load(os.path.join(dir_name,"clear_planes", fname))
    noised_plane = torch.load(os.path.join(dir_name,"noised_planes", fname))

    probs = torch.clone(clear_plane, device=device)
    z,_,_ = torch.where(probs==0)
    probs[probs==0] += probs[z].mean(-1).mean(-1)
    distr = torch.distributions.binomial.Binomial(total_count=num_trials, probs=torch.repeat_interleave(probs[None] ,total_crops,dim=0))
    n, x, y = clear_plane.shape
    c_x, c_y = crop_shape[0]//2, crop_shape[1]//2

    samples = torch.nonzero(distr.sample().to(device))

    l = torch.cumsum(torch.bincount(samples[:,0]*n+samples[:,1]), dim=0)
    l = torch.cat([torch.Tensor([0]).long().to(device),l])

    diff = l[1:]-l[:-1]
    base = l[:-1]

    sample = torch.rand(total_crops*n, device=device) * diff + base
    sample = samples[sample.long()]

    w_x = torch.arange(-c_x,c_x,device=device)
    w_y = torch.arange(-c_y,c_y,device=device)

    c_x = torch.Tensor([c_x],device=device).long()
    c_y = torch.Tensor([c_y],device=device).long()
    x = torch.Tensor([x],device=device).long()
    y = torch.Tensor([y],device=device).long()
    sample = (torch.min(torch.max(sample[:,-2],c_x), x-c_x), torch.min(torch.max(sample[:,-1],c_y), y-c_y))


    sample = (torch.arange(n).repeat(total_crops).reshape(-1,1,1),
            (sample[0][:,None] + w_x[None]).reshape(total_crops*n, -1, 1).to('cpu'),
            (sample[1][:,None] + w_y[None]).reshape(total_crops*n, 1, -1).to('cpu'))

    clear_plane[sample], noised_plane[sample]
    torch.save(clear_crops, os.path.join(dir_name,"clear_crops", fname))
    torch.save(noised_crops, os.path.join(dir_name,"noised_crops", fname))