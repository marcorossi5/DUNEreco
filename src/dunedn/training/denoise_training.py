# This file is part of DUNEdn by M. Rossi
"""
    This module contains the wrapper function for the ``dunedn train`` command.
"""
from pathlib import Path
from shutil import copyfile
from dunedn.training.dataloader import PlaneLoader, CropLoader
from dunedn.training.args import Args
from dunedn.training.train import train
from dunedn.networks.helpers import get_model_from_args
from dunedn.utils.utils import get_configcard_path, load_yaml


def add_arguments_training(parser):
    """
    Adds training subparser arguments.

    Parameters
    ----------
        - parser: ArgumentParser, training subparser object
    """
    parser.add_argument("configcard", type=Path, help="yaml configcard path")
    parser.add_argument("--output", type=Path, help="output folder", default=None)
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files if present"
    )
    parser.set_defaults(func=training)


def training(args):
    """
    Wrapper training function.

    Parameters
    ----------
        - args: NameSpace object, command line parsed arguments. It should
                contain configcard file name, output path and force boolean option.

    Returns
    -------
        - float, minimum loss over training
        - float, uncertainty over minimum loss
        - str, best checkpoint file name
    """
    return training_main(args.configcard, args.output, args.force)


def training_main(configcard, output, force):
    """
    Wrapper training function. Reads settings from configcard

    Parameters
    ----------
        - configcard: Path, path to the yaml configcard
        - output: Path, path to the output folder
        - force: bool, wether to overwrite existing output folder, if present

    Returns
    -------
        - float, minimum loss over training
        - float, uncertainty over minimum loss
        - str, best checkpoint file name
    """
    config_path = get_configcard_path(configcard)
    parameters = load_yaml(config_path)
    parameters.update(
        {"configcard": configcard, "output": output, "force": force, "rank": 0}
    )
    args = Args(**parameters)
    args.build_directories()
    copyfile(config_path, args.dir_output / "input_runcard.yaml")

    # create model
    model = get_model_from_args(args)

    # load datasets
    loader = PlaneLoader if args.model == "uscg" else CropLoader
    kwargs = (
        {} if args.model == "uscg" else {"crop_edge": args.crop_edge, "pct": args.pct}
    )
    train_data = loader(
        args.dataset_dir, "train", args.task, args.channel, args.threshold, **kwargs
    )
    if not args.model == "uscg":
        kwargs.pop("pct")
    val_data = PlaneLoader(
        args.dataset_dir, "val", args.task, args.channel, args.threshold, **kwargs
    )

    # train
    return train(args, train_data, val_data, model)
