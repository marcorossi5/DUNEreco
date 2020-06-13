import os
import glob

import numpy as np

from skimage.feature import canny

event_step = 15360
collection_step = 960
readout_step = 800
time_len = 6000
ada_step = event_step // (2*readout_step + collection_step)
#c_pedestal = 500
#r_pedestal = 1800

def normalize(planes):
    M = planes.max()
    if M==0:
        return planes, 0, M
    m = planes.min()
    return (planes - m)/(M-m), m, M

def load_files(path_clear, path_noise):
    clear_files = glob.glob(path_clear)
    noised_files = glob.glob(path_noise)

    for f_clear, f_noise in zip(clear_files, noised_files):
        clear_data = np.load(f_clear)[:, 2:]
        noised_data = np.load(f_noise)[:, 2:]
        yield clear_data, noised_data

def plane_idx():
    signal_planes = [i for i in range(ada_step)]

    readout = []
    collection = []
    cpp = 2*readout_step+collection_step #channel per plane

    for i in range(ada_step):
        readout.extend(range(cpp*i, cpp*i + 2*readout_step))
        collection.extend(range(cpp*i + 2*readout_step,
                               cpp*i + 2*readout_step + collection_step))

    return readout, collection

def stack_planes(clear_file, noised_file, r_idx, c_idx):
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
        r_n_clear.append(r_clear[i*readout_step:
                                                  (i+1)*readout_step])
        r_n_noised.append(r_noised[i*readout_step:
                                                  (i+1)*readout_step])

    for i in range(int(c_clear.shape[0]/collection_step)):
        if c_clear[i*collection_step:(i+1)*collection_step].max() == 0:
            print('skipped')
            continue
        c_n_clear.append(c_clear[i*collection_step:
                                                  (i+1)*collection_step])
        c_n_noised.append(c_noised[i*collection_step:
                                                    (i+1)*collection_step])

    return np.stack(r_n_clear), np.stack(r_n_noised),\
           np.stack(c_n_clear), np.stack(c_n_noised)

def get_planes(clear_file, noised_file):
    """
        Returning the three separate nonempty planes
    """
    r_idx, c_idx = plane_idx()

    return stack_planes(clear_file, noised_file, r_idx, c_idx)

def get_crop(clear_plane, n_crops=1000,
            crop_shape=(32,32), p=0.5):
    x, y = clear_plane.shape
    c_x, c_y = crop_shape[0]//2, crop_shape[1]//2

    #im = np.copy(clear_plane)
    #im[im!=0] = 1
    clear_plane = np.copy(clear_plane)
    im = canny(clear_plane).astype(float)

    sgn = np.transpose(np.where(im==1))
    bkg = np.transpose(np.where(im==0))

    samples = []
    sample = np.random.choice(len(sgn), size=int(n_crops*p))
    samples.append(sgn[sample])

    sample = np.random.choice(len(bkg), size=int(n_crops*(1-p)))
    samples.append(bkg[sample])

    samples = np.concatenate(samples)

    w = (np.minimum(np.maximum(samples[:,0], c_x), x-c_x),
        np.minimum(np.maximum(samples[:,1], c_y), y-c_y)) #crops centers

    return((w[0][:,None]+np.arange(-c_x,c_x)[None])[:,:,None],
           (w[1][:,None]+np.arange(-c_y,c_y)[None])[:,None,:])    