# This file is part of DUNEdn by M. Rossi
import numpy as np


def save_normalization_info(dir_name, channel):
    """
    Store on disk useful information to apply dataset normalization. Available
    normalizations are MinMax | Zscore | Mednorm

    Parameters
    ----------
        - dir_name: Path, directory path to datasets
        - channel: str, induction | collection
    """
    print(f"[+] Saving normalization info to {dir_name}")
    fname = dir_name / f"train/planes/{channel}_noisy.npy"
    n = np.load(fname).flatten()

    # MinMax
    fname = dir_name / f"{channel}_minmax"
    np.save(fname, [n.min(), n.max()])

    # Zscore
    fname = dir_name / f"{channel}_zscore"
    np.save(fname, [n.mean(), n.std()])

    # Mednorm
    medians = np.median(n.reshape([n.shape[0], -1]), axis=1)
    med_min = (n - medians).min()
    med_max = (n - medians).max()
    fname = dir_name / f"{channel}_mednorm"
    np.save(fname, [med_min, med_max])


def get_crop(clear_plane, nb_crops=1000, crop_size=(32, 32), pct=0.5):
    """
    Finds crops centers indeces and return crops around them.

    Parameters
    ----------
        - clear_plane: np.array, clear plane of shape=(H,W)
        - nb_crops: int, number of crops
        - crop_size: list, crop [height, width]
        - pct: float, signal / background crops balancing

    Returns
    -------
        - tuple, (idx_h, idx_w). idx_h of shape=(nb_crops, crop_edge, 1).
                 idx_w of shape=(nb_crops, 1, crop_edge).
    """
    x, y = clear_plane.shape
    c_x, c_y = crop_size[0] // 2, crop_size[1] // 2

    im = clear_plane != 0

    sgn = np.transpose(np.where(im == True))
    bkg = np.transpose(np.where(im == False))

    samples = []
    sample = np.random.choice(len(sgn), size=int(nb_crops * pct))
    samples.append(sgn[sample])

    sample = np.random.choice(len(bkg), size=int(nb_crops * (1 - pct)))
    samples.append(bkg[sample])

    samples = np.concatenate(samples)

    w = (
        np.minimum(np.maximum(samples[:, 0], c_x), x - c_x),
        np.minimum(np.maximum(samples[:, 1], c_y), y - c_y),
    )  # crops centers

    idx_h = (w[0][:, None] + np.arange(-c_x, c_x)[None])[:, :, None]
    idx_w = (w[1][:, None] + np.arange(-c_y, c_y)[None])[:, None, :]
    return (idx_h, idx_w)
