# This file is part of DUNEdn by M. Rossi
from pathlib import Path
from shutil import copyfile
from dunedn.denoising.dataloader import PlaneLoader, CropLoader
from dunedn.denoising.model import get_model_from_args
from dunedn.denoising.args import Args
from dunedn.denoising.train import train
from dunedn.utils.utils import get_runcard


def add_arguments_training(parser):
    parser.add_argument("runcard", type=Path, help="yaml configcard path")
    parser.add_argument("--output", type=Path, help="output folder", default=None)
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files if present"
    )
    parser.set_defaults(func=main_training)


def main_training(args):
    parameters = get_runcard(args.runcard)
    args = vars(args)
    args.pop("func")
    parameters.update(args)
    parameters["rank"] = 0
    args = Args(**parameters)
    args.build_directories()
    copyfile(args.runcard, args.dir_output / "input_runcard.yaml")

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
    kwargs.pop("pct")
    val_data = PlaneLoader(
        args.dataset_dir, "val", args.task, args.channel, args.threshold, **kwargs
    )

    # train
    return train(args, train_data, val_data, model)
