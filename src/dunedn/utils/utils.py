# This file is part of DUNEdn by M. Rossi
""" This module contains utility functions of general interest. """
import os
import logging
import yaml
import numpy as np
from hyperopt import hp
from dunedn.configdn import PACKAGE, get_dunedn_search_path


def check(check_instance, check_list):
    """
    Checks that check_list contains check_instance object. If not, raises
    NotImplementedError.

    Parameters
    ----------
        - check_instance: obj, object to check
        - check_list: list, available options

    Raises
    ------
        - NotImplementedError, if check_instance is not in check_list
    """
    if not check_instance in check_list:
        raise NotImplementedError("Operation not implemented")


def get_freer_gpu():
    """ Returns the gpu number with the most available memory. """
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmax(memory_available)


def get_freer_gpus(n):
    """ Returns the n gpus with the most available memory. """
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    np.argsort(memory_available)
    return np.argmax(memory_available)[-n:]


def smooth(smoothed, scalars, weight):  # weight between 0 and 1
    """ Computes the next moving average item. """
    assert len(scalars) - len(smoothed) == 1

    if len(scalars) == 1:
        smoothed.append(scalars[0])
    else:
        smoothed.append(weight * smoothed[-1] + (1 - weight) * scalars[-1])

    return smoothed


def moving_average(scalars, weight):
    """
    Computes the moving avarage from a list of scalar quantities.

    Parameters
    ----------
        - scalars: list, of scalar quantity to be soothed
        - weight: the weighting factor in the (0,1) range. Higher values provide
                  more smoothing power

    Returns
    -------
        - list, the averaged quantities
    """
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


# instantiate logger
logger = logging.getLogger(PACKAGE + ".train")


def load_yaml(runcard_file):
    """Loads yaml runcard"""
    with open(runcard_file, "r") as stream:
        runcard = yaml.load(stream, Loader=yaml.FullLoader)
    for key, value in runcard.items():
        if "hp." in str(value):
            logger.info("Evaluating: eval(%s) = %s", key, str(value))
            runcard[key] = eval(value)
    return runcard


def get_configcard_path(fname):
    """
    Retrieve the configcard path.
    If the supplied path is not a valid file, looks recursively into directories
    from DUNEDN_SEARCH_PATH environment variable to find the first match.

    Parameters
    ----------
        - fname: Path, path to configcard yaml file.

    Returns
    -------
        - Path, the retrieved configcard path

    Raises
    ------
        FileNotFoundError, if fname is not found.
    """
    if fname.is_file():
        return fname

    # get list of directories from DUNEDN_SEARCH_PATH env variable
    search_path = get_dunedn_search_path()

    # recursively look in search directories
    for base in search_path:
        candidate = base / fname.name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Configcard {fname} not found. Please, update DUNEDN_SEARCH_PATH variable."
    )


def print_summary_file(args):
    """Export Args object to file. """
    d = args.__dict__
    fname = args.dir_output / "readme.txt"
    with open(fname, "w") as f:
        f.writelines("Model summary file:\n")
        for k in d.keys():
            f.writelines("\n%s     %s" % (str(k), str(d[k])))
        f.close()
