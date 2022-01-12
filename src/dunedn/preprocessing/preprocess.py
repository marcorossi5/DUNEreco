# This file is part of DUNEdn by M. Rossi
"""
    This module contains the wrapper function for the ``dunedn preprocess``
    command.
"""
from pathlib import Path
from dunedn.utils.utils import get_configcard_path, load_yaml
from dunedn.preprocessing.putils import (
    get_planes_and_dump,
    save_normalization_info,
    crop_planes_and_dump,
)


def add_arguments_preprocessing(parser):
    """
    Adds preprocessing subparser arguments.

    Parameters
    ----------
        - parser: ArgumentParser, preprocessing subparser object
    """
    parser.add_argument("configcard", type=Path, help="yaml configcard path")
    parser.add_argument(
        "--save_sample", action="store_true", help="extract a smaller dataset"
    )
    parser.set_defaults(func=preprocess)


def preprocess(args):
    """
    Wrapper preprocessing function.

    Parameters
    ----------
        - args: NameSpace object, command line parsed arguments. It should
                contain configcard file name, dataset directory path, plus
                save_sample boolean options.
    """
    config_path = get_configcard_path(args.configcard)
    params = load_yaml(config_path)
    preprocess_main(
        Path(params["dataset_dir"]),
        params["nb_crops"],
        params["crop_edge"],
        params["pct"],
        args.save_sample,
    )


def preprocess_main(dir_name, nb_crops, crop_edge, pct, save_sample):
    """
    Preprocessing main function. Loads an input event from file, makes inference and
    saves the ouptut. Eventually returns the output array.

    Parameters
    ----------
        - dir_name: Path, directory path to dataset
        - nb_crops: int, number of crops from each plane
        - crop_edge: int, crop edge size
        - pct: float, signal / background crop balance
        - save_sample: bool, wether to extract a smaller dataset

    Returns
    -------
        - np.array, ouptut event of shape=(nb wires, nb tdc ticks)
    """
    crop_size = (crop_edge, crop_edge)
    for folder in ["train", "val", "test"]:
        dname = dir_name / folder
        (dname / "planes").mkdir(parents=True, exist_ok=True)
        if folder == "train":
            (dname / "crops").mkdir(exist_ok=True)
        get_planes_and_dump(dname, save_sample)
    for channel in ["induction", "collection"]:
        save_normalization_info(dir_name, channel)
    crop_planes_and_dump(dir_name / "train", nb_crops, crop_size, pct)
