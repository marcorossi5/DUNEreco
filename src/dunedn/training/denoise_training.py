"""
    This module contains the wrapper function for the ``dunedn train`` command.

    Example
    -------

    Train help output:

    .. code-block:: text

        $ dunedn train --help

        usage: dunedn train [-h] [--model {cnn,gcnn,uscg}] [--output OUTPUT] [--force] [--interactive]

        Train model loading settings from configcard.

        optional arguments:
          -h, --help            show this help message and exit
          --model {cnn,gcnn,uscg}, -m {cnn,gcnn,uscg}
                                the model to train
          --output OUTPUT, -o OUTPUT
                                output folder
          --force               overwrite existing files if present
          --interactive, -i     triggers interactive mode
"""
import logging
from pathlib import Path
from dunedn import PACKAGE
from dunedn.utils.ask_edit_card import ask_edit_card
from dunedn.utils.utils import (
    load_runcard,
    check_in_folder,
    initialize_output_folder,
    save_runcard,
)

logger = logging.getLogger(PACKAGE + ".training")


def add_arguments_training(parser):
    """
    Adds training subparser arguments.

    Parameters
    ----------
    parser: ArgumentParser
        Training subparser object.
    """
    valid_models = ["cnn", "gcnn", "uscg", "performer"]
    parser.add_argument(
        "--model", "-m", help="the model to train", choices=valid_models
    )
    parser.add_argument("--output", "-o", type=Path, help="output folder", default=None)
    parser.add_argument(
        "--force", action="store_true", help="overwrite existing files if present"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="triggers interactive mode"
    )
    parser.add_argument(
        "--runcard",
        type=Path,
        help="uses runcard and initializes new output folder from scratch",
        default=None,
    )
    parser.set_defaults(func=training)


def training(args):
    """Wrapper training function.

    Parameters
    ----------
    args: NameSpace
        Command line parsed arguments.

    Returns
    -------
    float
        Minimum loss over training.
    float
        Uncertainty over minimum loss.
    str
        Best checkpoint file name.
    """
    # validate `runcard` argument
    if args.runcard and args.interactive:
        logger.error("Argument `--runcard` cannot be used with interactive mode")
        exit(-1)
    if args.interactive:
        ask_edit_card(logger, args.output)
    runcard_path = (
        args.runcard if args.runcard is not None else args.output / "cards/runcard.yaml"
    )
    # load runcard and setup output folder structure
    setup = load_runcard(runcard_path)

    if args.runcard:
        initialize_output_folder(args.output, args.force)
        setup["output"] = args.output
        save_runcard(args.output / "cards/runcard.yaml", setup)
        # save a default runcard in folder to allow default resoration
        save_runcard(args.output / "cards/runcard_default.yaml", setup)

    check_in_folder(setup["output"] / f"models/{args.model}", args.force)

    # launch main training function
    training_main(args.model, setup)


def training_main(modeltype: str, setup: dict):
    """Main training function.

    Parameters
    ----------
    modeltype: str
        The model to be trained. Available options: cnn | gcnn | uscg.
    setup: dict
        Settings dictionary.
    """
    from dunedn.networks.gcnn.training import gcnn_training
    from dunedn.networks.uscg.training import uscg_training
    from dunedn.networks.performer.training import performer_training

    if modeltype in ["cnn", "gcnn"]:
        logger.info(f"Training {modeltype} network")
        gcnn_training(modeltype, setup)
    elif modeltype == "uscg":
        logger.info("Training USCG network")
        uscg_training(setup)
    elif modeltype == "performer":
        logger.info("Training performer network")
        performer_training(setup)
    else:
        raise NotImplementedError(f"model not implemented, found: {modeltype}")
