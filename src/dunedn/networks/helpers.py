# This file is part of DUNEdn by M. Rossi
"""
    This module contains the helper functions to retrieve a model from its model
    name.
"""

from dunedn.networks.models import USCG_Net, GCNN_Net
from dunedn.networks.model_utils import MyDataParallel


supported_models = ["uscg", "cnn", "gcnn"]


def get_supported_models():
    """
    Gets the names of the supported models.

    Returns
    -------
        - list, the list of currently implemented models
    """
    return supported_models


def get_model(modeltype, **args):
    """
    Utility function to retrieve model from model name and args.

    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn | uscg
        - args: list, model's __init__ arguments

    Returns
    -------
        - torch.nn.Module, the model instance

    Raises
    ------
        - NotImplementedError if modeltype is not in ['uscg', 'cnn', 'gcnn']
    """
    if modeltype == "uscg":
        return USCG_Net(**args)
    elif modeltype in ["gcnn", "cnn"]:
        return GCNN_Net(**args)
    else:
        raise NotImplementedError("Model not implemented")


def get_model_from_args(args):
    """
    Load model from argument object. The arguments' model attribute contains the
    name of the network to be loaded.

    Model independent attributes:
        - model: str, available options cnn | gcnn
        - dev: str, device hosting computation

    Model dependent attributes.

    args.model is 'uscg'
    Attributes:
        - model: str, available options cnn | gcnn
        - dev: str, device hosting computation
        - task: str, available options dn | roi
        - h: int, input image height
        - w: int, input image width

    args.model is one of 'cnn' | 'gcnn'
    Attributes:
        - task: str, available options dn | roi
        - crop_edge: int, crop edge size
        - input_channels: int, inputh channel dimension size
        - hidden_channels: int, convolutions hidden filters number
        - k: int, nearest neighbor number. None if model is cnn

    Parameters
    ----------
        - args: Args, runtime settings

    Returns
    -------
        - MyDataParallel, the loaded model

    Raises
    ------
        - NotImplementedError if modeltype is not in ['uscg', 'cnn', 'gcnn']
    """
    kwargs = {}
    if args.model == "uscg":
        kwargs["task"] = args.task
        kwargs["h"] = args.patch_h
        kwargs["w"] = args.patch_w
    elif args.model in ["cnn", "gcnn"]:
        kwargs["model"] = args.model
        kwargs["task"] = args.task
        kwargs["crop_edge"] = args.crop_edge
        kwargs["input_channels"] = args.input_channels
        kwargs["hidden_channels"] = args.hidden_channels
        kwargs["k"] = args.k if args.model == "gcnn" else None
    else:
        raise NotImplementedError("Loss function not implemented")
    return MyDataParallel(get_model(args.model, **kwargs), device_ids=args.dev)
