"""This module implements utility function for all the networks."""
from logging import Logger
import numpy as np

supported_models = ["uscg", "cnn", "gcnn"]


def get_supported_models():
    """
    Gets the names of the supported models.

    Returns
    -------
        - list, the list of currently implemented models
    """
    return supported_models


def get_hits_from_clear_images(planes: np.ndarray, threshold: float) -> np.ndarray:
    """Segment input images as signal-background pixels.

    Parameters
    ----------
    planes: np.ndarray
        The clear planes, of shape=(N,1,H,W).
    threshold: float
        Threshold above which a pixel is considered containing signal.

    Returns
    -------
    hits: np.ndarray
        The signal-background segmented image, of shape=(N,1,C,W).
    """
    mask = np.abs(planes) >= threshold
    hits = np.zeros_like(planes)
    hits[mask] = 1
    return hits


def apply_median_subtraction(planes: np.ndarray) -> np.ndarray:
    """Computes median subtraction to input planes.

    Parameters
    ----------
    planes: np.ndarray
        The data to be normalized.

    Returns
    -------
    output: np.ndarray
        The median subtracted data.
    """
    medians = np.median(planes, axis=[1, 2, 3], keepdims=True)
    output = planes - medians
    return output


def print_epoch_logs(logger: Logger, metrics_names: list[str], logs: dict):
    """Prints logs dictionary on epoch end.

    Parameters
    ----------
    logger: Logger
        The logging object.
    metrics_names: list[str]
        The list of metrics to be printed.
    logs: dict
        The computed metrics values to be logged.
    """
    msg = f"Took {logs['epoch_time']:.0f} s, "
    all_metrics_names = metrics_names + [f"val_{name}" for name in metrics_names]
    for name in all_metrics_names:
        mean = logs.get(name)
        std = logs.get(f"{name}_std")
        if mean is not None and std is not None:
            msg += f"{name}: {mean:.3f} +/- {std:.3f}, "
    msg = msg.strip(" ,")
    logger.info(msg)


def print_cfnm(cfnm, channel):
    """Prints confusion matrix.

    Parameters
    ----------
    cfnm: list
        Computed confusion matrix.
    channel: str
        Available options readout | collection.

    Returns
    -------
    msg: str
        The confusion matrix representatiton to be printed.
    """
    tp, fp, fn, tn = cfnm
    msg = (
        f"Confusion Matrix on {channel} planes:\n"
        f"\tTrue positives: {tp[0]:.3f} +- {tp[1]:.3f}\n"
        f"\tTrue negatives: {tn[0]:.3f} +- {tn[1]:.3f}\n"
        f"\tFalse positives: {fp[0]:.3f} +- {fp[1]:.3f}\n"
        f"\tFalse negatives: {fn[0]:.3f} +- {fn[1]:.3f}"
    )
    return msg
