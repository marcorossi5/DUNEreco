""" This module provides functions for USCG network training and loading."""
import logging
from pathlib import Path
import torch
from .performer_dataloading import PlanesDataset
from .performer_net import PerformerNet
from dunedn import PACKAGE
from dunedn.training.metrics import DN_METRICS
from dunedn.training.losses import get_loss

logger = logging.getLogger(PACKAGE + ".performer")


def load_and_compile_performer_network(
    msetup: dict, checkpoint_filepath: Path = None
) -> PerformerNet:
    """Loads a USCG network.

    Parameters
    ---------
    msetup: dict
        The model setup dictionary.
    checkpoint_filepath: Path
        The `.pth` checkpoint containing network weights to be loaded.

    Returns
    -------
    network: UscgNet
        The loaded neural network.
    """
    network = PerformerNet(**msetup["net_dict"])

    if checkpoint_filepath:
        logger.info(f"Loading weights at {checkpoint_filepath}")
        state_dict = torch.load(checkpoint_filepath, map_location=torch.device("cpu"))
        network.load_state_dict(state_dict)

    # loss
    loss = get_loss(msetup["loss_fn"])()

    # optimizer
    optimizer = torch.optim.Adam(list(network.parameters()), float(msetup["lr"]))

    network.compile(loss, optimizer, DN_METRICS)

    return network


def performer_training(setup: dict):
    """Performer network training.

    Parameters
    ----------
    setup: dict
        Settings dictionary.
    """
    msetup = setup["model"]["performer"]
    # model loading
    network = load_and_compile_performer_network(msetup, msetup["ckpt"])

    # TODO: remove channel (collection | induction) hard coding
    # data loading
    gen_kwargs = {
        "task": setup["task"],
        "dsetup": setup["dataset"],
    }
    data_folder = setup["dataset"]["data_folder"]
    train_igenerator = PlanesDataset(
        data_folder / "train/planes/induction_noisy.npy", should_load_target=True
    )
    train_cgenerator = PlanesDataset(
        data_folder / "train/planes/collection_noisy.npy", should_load_target=True
    )
    val_igenerator = PlanesDataset(
        data_folder / "val/planes/induction_noisy.npy", should_load_target=True
    )
    val_cgenerator = PlanesDataset(
        data_folder / "val/planes/collection_noisy.npy", should_load_target=True
    )
    test_igenerator = PlanesDataset(
        data_folder / "test/planes/induction_noisy.npy", should_load_target=True
    )
    test_cgenerator = PlanesDataset(
        data_folder / "test/planes/collection_noisy.npy", should_load_target=True
    )

    train_generators = (train_igenerator, train_cgenerator)
    val_generators = (val_igenerator, val_cgenerator)
    test_generators = (test_igenerator, test_cgenerator)

    # training
    network.fit(
        train_generators,
        epochs=setup["model"]["epochs"],
        val_generator=val_generators,
        dev=setup["dev"],
    )

    # testing
    _, logs = network.predict(test_generators)
    network.metrics_list.print_metrics(logger, logs)
