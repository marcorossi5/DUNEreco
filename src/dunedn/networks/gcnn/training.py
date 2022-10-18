""" This module provides functions for CNN and GCNN networks training and loading."""
import logging
from pathlib import Path
import torch
from .gcnn_dataloading import TilingDataset
from .gcnn_net import GcnnNet
from dunedn import PACKAGE
from dunedn.networks.onnx.onnx_gcnn_net import OnnxGcnnNetwork
from dunedn.training.callbacks import ModelCheckpoint
from dunedn.training.losses import get_loss
from dunedn.training.metrics import DN_METRICS

logger = logging.getLogger(PACKAGE + ".gcnn")


def load_and_compile_gcnn_onnx_network(checkpoint_path: Path):
    logger.info(f"Loading Gcnn ONNX network weights at {checkpoint_path}")
    return OnnxGcnnNetwork(
        checkpoint_path.as_posix(),
        DN_METRICS,
        # providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )


def load_and_compile_gcnn_network(
    msetup: dict, checkpoint_filepath: Path = None
) -> GcnnNet:
    """Loads CNN or GCNN network.

    Parameters
    ----------
    msetup: dict
        The model setup dictionary, which should contain:

        - ``net_dict``: nested dictionary with ``input_channels``,
          ``hidden_channels``, ``nb_lpfs``, ``k``, ``name``
        - ``loss_fn`` (optional): the objective function name for training purposes
        - ``lr`` (optional): the learning rate for training purposes

    checkpoint_filepath: Path
        The `.pth` checkpoint containing network weights to be loaded.

    Returns
    -------
    network: GcnnNet
        The loaded neural network.
    """
    network = GcnnNet(**msetup["net_dict"])

    if checkpoint_filepath:
        logger.info(f"Loading Gcnn network weights at {checkpoint_filepath}")
        state_dict = torch.load(checkpoint_filepath, map_location=torch.device("cpu"))
        network.load_state_dict(state_dict)

    # loss
    loss_name = msetup.get("loss_fn")
    if loss_name is None:
        loss_name = "mse"
    loss = get_loss(loss_name)()

    # optimizer
    lr = msetup.get("lr")
    if lr is None:
        lr = "1e-3"
    optimizer = torch.optim.Adam(list(network.parameters()), float(lr))
    network.compile(loss, optimizer, DN_METRICS)

    return network


def gcnn_training(modeltype: str, setup: dict):
    """GCNN network training.

    Parameters
    ----------
    modeltype: str
        The model to be trained. Available options: cnn | gcnn.
    setup: dict
        Settings dictionary.
    """
    # model loading
    assert modeltype in ["cnn", "gcnn"]
    msetup = setup["model"][modeltype]
    network = load_and_compile_gcnn_network(msetup, msetup["ckpt"])

    # TODO: remove channel (collection | induction) hard coding
    # data loading
    data_folder = setup["dataset"]["data_folder"]

    train_folder = data_folder / "train/evts"
    val_folder = data_folder / "val/evts"

    logger.info(f"Loading training dataset from {train_folder}")
    train_generator = TilingDataset(
        train_folder,
        batch_size=msetup["batch_size"],
        crop_size=msetup["crop_size"],
        has_target=True,
    )

    logger.info(f"Loading training dataset from {val_folder}")
    val_generator = TilingDataset(
        val_folder,
        batch_size=msetup["batch_size"],
        crop_size=msetup["crop_size"],
        has_target=True,
    )

    # training
    callbacks = [
        ModelCheckpoint(
            setup["output"] / "models" / modeltype / f"{modeltype}.pth",
            "val_psnr",
            save_best_only=True,
            mode="max",
            verbose=1,
        )
    ]
    network.fit(
        train_generator,
        epochs=setup["model"]["epochs"],
        val_generator=val_generator,
        dev=setup["dev"],
        callbacks=callbacks,
    )

    # testing
    test_folder = data_folder / "test/evts"
    logger.info("Stop training, now testing")
    logger.info(f"Loading training dataset from {test_folder}")
    test_generator = TilingDataset(
        test_folder,
        batch_size=msetup["batch_size"],
        crop_size=msetup["crop_size"],
        has_target=True,
    )
    _, logs = network.predict(test_generator)
    network.metrics_list.print_metrics(logger, logs)
