# This file is part of DUNEdn by M. Rossi
import os
import numpy as np
import yaml
from hyperopt import hp


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def get_freer_gpus(n):
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    ind = np.argsort(memory_available)
    return np.argmax(memory_available)[-n:]


def smooth(smoothed, scalars, weight):  # weight between 0 and 1
    assert len(scalars) - len(smoothed) == 1

    if len(scalars) == 1:
        smoothed.append(scalars[0])
    else:
        smoothed.append(weight * smoothed[-1] + (1 - weight) * scalars[-1])

    return smoothed


def moving_average(scalars, weight):
    smoothed = []
    for i in range(len(scalars)):
        smooth(smoothed, scalars[: i + 1], weight)
    return smoothed


def median_subtraction(planes):
    """
    Subtracts median value from input planes.

    Parameters
    ----------
        planes: np.array, array of shape=(N,C,H,W)

    Returns
    -------
        -np.array, median subtracted planes of shape=(N,C,H,W)
    """
    shape = [planes.shape[0], -1]
    medians = np.median(planes.reshape(shape), axis=1)
    return planes - medians[:, None, None, None]


def confusion_matrix(hit, no_hit, t=0.5):
    """
    Return confusion matrix elements from arrays of scores and threshold value.

    Parameters:
        hit: np.array, scores of real hits
        no_hit: np.array, scores of real no-hits
        t: float, threshold
    Returns:
        tp, fp, fn, tn
    """
    tp = np.count_nonzero(hit > t)
    fn = np.size(hit) - tp

    tn = np.count_nonzero(no_hit < t)
    fp = np.size(no_hit) - tn

    return tp, fp, fn, tn


def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    for key, value in runcard.items():
        if ("hp." in str(value)) or ("None" == str(value)):
            runcard[key] = eval(value)
    return runcard
