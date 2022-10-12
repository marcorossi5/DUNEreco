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
    """Loads a Performer network.

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
    from dunedn.networks.gcnn.gcnn_dataloading import TilingDataset

    train_generator = TilingDataset(
        data_folder / "train/evts/rawdigit_evt0.npy",
        batch_size=msetup["batch_size"],
        crop_size=msetup["crop_size"],
        has_target=True,
    )
    print(train_generator.clear_crops.shape, train_generator.noisy_crops.shape)
    print(train_generator.nb_icrops, train_generator.nb_ccrops)
    exit()
    # train_generator = PlanesDataset(
    #     data_folder / "train/planes", setup["model"]["performer"]["batch_size"], has_target=True
    # )
    val_generator = None
    test_generator = None
    # val_generator = PlanesDataset(
    #     data_folder / "val/planes", setup["model"]["performer"]["batch_size"], has_target=True
    # )
    # test_generator = PlanesDataset(
    #     data_folder / "test/planes", setup["model"]["performer"]["batch_size"], has_target=True
    # )

    # logger.info(
    #     f"Dataset loaded: train / validate / test on "
    #     f"{len(train_generator)} / {len(val_generator)} / {len(test_generator)} events"
    # )

    # training
    network.fit(
        train_generator,
        epochs=setup["model"]["epochs"],
        val_generator=val_generator,
        dev=setup["dev"],
    )

    # testing
    _, logs = network.predict(test_generator)
    network.metrics_list.print_metrics(logger, logs)
