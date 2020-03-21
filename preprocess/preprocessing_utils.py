import numpy as np
import glob
event_step = 15360
collection_step = 960
readout_step = 800
time_len = 6000
ada_step = event_step // (readout_step*2 + collection_step)
import tqdm


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
        c_n_clear.append(normalize(c_clear[i*collection_step :(i+1)*collection_step]))
        c_n_noised.append(normalize(c_noised[i*collection_step:(i+1)*collection_step]))

    return np.stack(r_n_clear), np.stack(r_n_noised), np.stack(c_n_clear), np.stack(c_n_noised)

def get_planes(clear_file, noised_file):
    """
        Returning the three separate nonempty planes
    """
    r_idx, c_idx = plane_idx()

    return normalize_planes(clear_file, noised_file, r_idx, c_idx)

def get_crop(clear_plane, noised_plane, total_crops=1000, crop_shape=(32,32), num_trials=5):
    probs = np.copy(clear_plane)
    z,_,_ = np.where(probs==0)
    probs[probs==0] += probs[z].mean(-1).mean(-1)
    n, x, y = clear_plane.shape
    c_x, c_y = crop_shape[0]//2, crop_shape[1]//2

    for i in range(total_crops):
        samples = np.argwhere(sample_binomial(num_trials, probs))
        count = 0
        while len(np.where(np.bincount(samples[:,0])==0)[0]) > 1:
            print('Repeating sampling time number', count)
            count += 1
            samples = np.argwhere(sample_binomial(num_trials, probs))
        
        l = np.cumsum(np.insert(np.bincount(samples[:,0]),0,0))

        sample = []
        for j in range(n):
            sample.append(np.random.randint(l[j],l[j+1]))
        sample = samples[sample]

        sample = (np.minimum(np.maximum(sample[:,1],c_x), x-c_x), np.minimum(np.maximum(sample[:,2],c_y), y-c_y))
        w_x = np.arange(-c_x,c_x)
        w_y = np.arange(-c_y,c_y)
        sample = (np.arange(n).reshape([-1,1,1]),
                (sample[0][:,None] + w_x[None])[:,None],
                (sample[1][:,None] + w_y[None])[:,:,None])
        clear_crop = clear_plane[sample]
        noised_crop = noised_plane[sample]
        yield clear_crop, noised_crop