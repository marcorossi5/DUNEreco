import numpy as np
import glob
event_step = 15360
ada_step = 6
collection_step = 960
readout_step = 800
time_len = 6000
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

def get_planes(clear_file, noised_file):
    """
        Returning the three separate nonempty planes
    """
    signal_planes = [i for i in range(ada_step)]

    cpp = 2*readout_step+collection_step #channel per plane
    steps = [0,readout_step, 2*readout_step, 2*readout_step+collection_step]


    for i in signal_planes:
        for j in range(len(steps)-1):
            clear_plane = clear_file[i*cpp+steps[j]:
                                i*cpp+steps[j+1]]
            if clear_plane.max() == 0:
                continue
            noised_plane = noised_file[i*cpp+steps[j]:
                                            i*cpp+steps[j+1]]
            clear_plane = normalize(clear_plane)
            noised_plane = normalize(noised_plane)
            #pad readout planes with np.inf to build a stackable array
            if clear_plane.shape[0] == readout_step:
                clear_plane = np.pad(clear_plane, ((0,collection_step-readout_step), (0,0)), 'constant',constant_values=np.inf)
                noised_plane = np.pad(noised_plane, ((0,collection_step-readout_step), (0,0)), 'constant',constant_values=np.inf)
            yield clear_plane, noised_plane

def get_crop(clear_plane, noised_plane, total_crops=1000, crop_shape=(32,32), num_trials=5):
    probs = np.copy(clear_plane)
    probs[probs==0] += probs.mean()
    x, y = clear_plane.shape
    c_x, c_y = crop_shape[0]//2, crop_shape[1]//2

    for i in range(total_crops):
        samples = np.transpose(np.nonzero(sample_binomial(num_trials, probs)))
        while len(samples[0]) < 1:
            samples = np.transpose(np.nonzero(sample_binomial(num_trials, probs)))
        sample = np.random.choice(len(samples))
        sample = samples[sample]

        sample = (min(max(int(sample[0]), c_x), x-c_x), min(max(int(sample[1]), c_y),y-c_y))
        clear_crop = clear_plane[sample[0]-c_x:sample[0]+c_x, sample[1]-c_y:sample[1]+c_y]
        noised_crop = noised_plane[sample[0]-c_x:sample[0]+c_x, sample[1]-c_y:sample[1]+c_y]
        yield clear_crop, noised_crop