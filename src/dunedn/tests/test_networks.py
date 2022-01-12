"""
    Ensures DUNEdn networks objects run the forwrd pass without errors.
"""
import logging
import torch
from collections import namedtuple
from dunedn.configdn import PACKAGE
from dunedn.networks.helpers import get_supported_models, get_model_from_args

# instantiate logger
logger = logging.getLogger(PACKAGE + ".test")

# namedtuple for uscg arguments
UscgArgsTuple = namedtuple("Args", ["model", "dev", "task", "patch_h", "patch_w"])

# namedtuple for gcnn arguments
GcnnArgsTuple = namedtuple(
    "Args",
    ["model", "dev", "task", "crop_edge", "input_channels", "hidden_channels", "k"],
)


def run_test_uscg():
    """ Run USCG network test. """
    # tuple containing induction and collection inference arguments
    batch_size = 32
    patch_h = 100
    patch_w = 160

    # load dummy dataset
    dummy_dataset = torch.rand(batch_size, 1, patch_h, patch_w)

    # load cnn model
    args = UscgArgsTuple("uscg", "cpu", "dn", patch_h, patch_w)
    model = get_model_from_args(args)
    model.eval()

    # forward pass
    output = model(dummy_dataset)

    # check that input and output have the same shape
    try:
        assert dummy_dataset.shape == output.shape
    except AssertionError as err:
        logger.critical(
            "Assertion fail: uscg model input and output shapes do not match"
        )
        raise err


def run_test_gcnn(modeltype):
    """
    Run GCNN-like network test.

    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn
    """
    # tuple containing induction and collection inference arguments
    batch_size = 32
    crop_edge = 32
    input_channels = 1
    k = 8 if modeltype == "gcnn" else None

    # load dummy dataset
    dummy_dataset = torch.rand(batch_size, input_channels, crop_edge, crop_edge)

    # load cnn model
    args = GcnnArgsTuple(modeltype, "cpu", "dn", crop_edge, input_channels, 32, k)
    model = get_model_from_args(args)
    model.eval()

    # forward pass
    output = model(dummy_dataset)

    # check that input and output have the same shape
    try:
        assert dummy_dataset.shape == output.shape
    except AssertionError as err:
        logger.critical(
            f"Assertion fail: {model} model input and output shapes do not match"
        )
        raise err


def run_test(modeltype):
    """
    Run the appropriate test for the supported model.

    Parameters
    ----------
        - modeltype: str, available options uscg | cnn | gcnn
    """
    logger.info(f"Running forward-pass test on {modeltype} model")
    if modeltype == "uscg":
        run_test_uscg()
    elif modeltype in ["cnn", "gcnn"]:
        run_test_gcnn(modeltype)


def test_networks():
    for modeltype in get_supported_models():
        run_test(modeltype)


if __name__ == "__main__":
    test_networks()
