# This file is part of DUNEdn by M. Rossi
"""
    This module contains the helper functions to retrieve a model from its model
    name.
"""

from dunedn.networks.models import USCG_Net, GCNN_Net
from dunedn.networks.model_utils import MyDataParallel


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
    Load model from args.

    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn | uscg
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
