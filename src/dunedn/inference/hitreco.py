# This file is part of DUNEdn by M. Rossi
"""
    This module contains utility functions for the inference step.
"""
import logging
import collections
from math import sqrt
import torch
from torch.utils.data import DataLoader
from dunedn.configdn import PACKAGE
from dunedn.training.args import Args
from dunedn.networks.helpers import get_model_from_args
from dunedn.networks.model_utils import model2batch
from dunedn.networks.GCNN_Net_utils import Converter
from dunedn.training.dataloader import InferenceLoader, InferenceCropLoader
from dunedn.training.train import inference, identity_inference, gcnn_inference
from dunedn.training.losses import get_loss
from dunedn.utils.utils import load_yaml, median_subtraction, get_configcard_path
from dunedn.geometry.helpers import evt2planes, planes2evt

# instantiate logger
logger = logging.getLogger(PACKAGE + ".inference")

# tuple containing induction and collection models
ModelTuple = collections.namedtuple("Model", ["induction", "collection"])

# tuple containing induction and collection inference arguments
ArgsTuple = collections.namedtuple("Args", ["batch_size", "patch_stride", "crop_size"])

task_dict = {"dn": "Denoising", "ROI": "Region of interest selection"}


def get_model_and_args(modeltype, task, channel, ckpt=None):
    """
    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn | uscg
        - task: str, available options dn | roi
        - channel: str, available options readout | collection
        - ckpt: Path, path to directory with saved model

    Returns
    -------
        - ArgsTuple, tuple containing induction and collection inference arguments
        - MyDataParallel, the loaded model
    """
    card = f"{modeltype}_{task}_{channel}_config.yaml"
    config_path = get_configcard_path(card)
    params = load_yaml(config_path)
    params["channel"] = channel
    args = Args(**params)

    crop_size = None if modeltype == "uscg" else args.crop_size
    patch_stride = args.patch_stride if modeltype == "uscg" else None
    batch_size = model2batch[modeltype][task]

    model = get_model_from_args(args)

    if ckpt is not None:
        fname = ckpt / f"{channel}/{ckpt.name}_{task}_{channel}.pth"
        state_dict = torch.load(fname)
        model.load_state_dict(state_dict)
    return ArgsTuple(batch_size, patch_stride, crop_size), model


def mkModel(modeltype, task, ckpt=None):
    """
    Instantiate a new model of type modeltype.

    Parameters
    ----------
        - modeltype: str, valid options: "uscg" | "cnn" | "gcnn" | "id"
        - task: str, valid options: "dn" | "roi"
        - ckpt: Path, checkpoint path

    Returns
    -------
        - list, of arguments to call model.inference for induction and collection
                respectively
        - ModelTuple, induction and collection models instances
    """
    if modeltype == "id":
        return [None, None], ModelTuple(None, None)
    iargs, imodel = get_model_and_args(modeltype, task, "induction", ckpt)
    cargs, cmodel = get_model_and_args(modeltype, task, "collection", ckpt)
    return [iargs, cargs], ModelTuple(imodel, cmodel)


def _uscg_inference(planes, loader, model, args, dev):
    """
    USCG inference utility function.

    Parameters
    ----------
        - planes: np.array, planes array of shape=(N,C,H,W)
        - loader: InferenceLoader, data loader
        - model: USCG_Net, network instance
        - args: ArgsTuple, inference arguments
        - dev: list | str, host device

    Returns
    -------
        - torch.Tensor, output tensor of shape=(N,C,H,W)
    """
    dataset = loader(planes)
    test = DataLoader(dataset=dataset, batch_size=args.batch_size)
    return inference(test, args.patch_stride, model.to(dev), dev).cpu()


def _gcnn_inference(planes, loader, model, args, dev):
    """
    GCNN inference utility function.

    Parameters
    ----------
        - planes: np.array, planes array of shape=(N,C,H,W)
        - loader: InferenceCropLoader, data loader
        - model: GCNN_Net, network instance
        - args: ArgsTuple, inference arguments
        - dev: list | str, host device

    Returns
    -------
        - torch.Tensor, output tensor of shape=(N,C,H,W)
    """
    # creating a new instance of converter every time could waste time if the
    # inference is called many times.
    # TODO: think about to make it a DnRoiModel attribute and pass it to the fn
    # TODO: the batch size changes according to task, modeltype
    sub_planes = torch.Tensor(median_subtraction(planes))
    converter = Converter(args.crop_size)
    tiles = converter.planes2tiles(sub_planes)

    dataset = loader(tiles)
    test = DataLoader(dataset=dataset, batch_size=args.batch_size)
    res = gcnn_inference(test, model.to(dev), dev).cpu()
    return converter.tiles2planes(res)


def _identity_inference(planes, loader, **kwargs):
    """
    USCG inference utility function.

    Parameters
    ----------
        - planes: np.array, planes array of shape=(N,C,H,W)
        - loader: torch.utils.data.DataLoader, data loader
        - kwargs: dict, kwargs for consistency with other inference functions

    Returns
    -------
        - torch.Tensor, output tensor of shape=(N,C,H,W)
    """
    dataset = loader(planes)
    test = DataLoader(dataset=dataset)
    return identity_inference(test).cpu()


def get_inference(modeltype, **kwargs):
    """
    Utility function to retrieve inference from model name and args.

    Parameters
    ----------
        - modeltype: str, available options cnn | gcnn | uscg
        - kwargs: dict, inference kwargs

    Returns
    -------
        - inference function

    Raises
    ------
        - NotImplementedError if modeltype is not in ['uscg', 'cnn', 'gcnn']
    """
    if modeltype == "uscg":
        return _uscg_inference(**kwargs)
    elif modeltype in ["cnn", "gcnn"]:
        return _gcnn_inference(**kwargs)
    elif modeltype == "id":
        return _identity_inference(**kwargs)
    else:
        raise NotImplementedError("Inference function not implemented")


class BaseModel:
    """
    Mother class for inference model.
    """

    def __init__(self, modeltype, task, ckpt=None):
        """
        Parameters
        ----------
            - modeltype: str, available options cnn | gcnn | uscg
            - task: str, available options dn | roi
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        self.modeltype = modeltype
        self.args, self.model = mkModel(modeltype, task, ckpt)
        self.loader = InferenceLoader if modeltype == "uscg" else InferenceCropLoader

    def inference(self, event, dev):
        """
        Interface for roi selection inference on a complete event.

        Parameters
        ----------
            - event: array-like, event input array of shape=(nb wires, nb tdc ticks)
            - dev: str, device hosting the computation

        Returns
        -------
            - np.array, denoised event of shape=(nb wires, nb tdc ticks)
        """
        inductions, collections = evt2planes(event)
        iout = get_inference(
            self.modeltype,
            planes=inductions,
            loader=self.loader,
            model=self.model.induction,
            args=self.args[0],
            dev=dev,
        )
        cout = get_inference(
            self.modeltype,
            planes=collections,
            loader=self.loader,
            model=self.model.collection,
            args=self.args[1],
            dev=dev,
        )
        # TODO: for the denoising model
        # masking for gcnn output must be done
        # think how to pass out the norm variables
        # probably the model itself is not correct in the current version
        # if self.modeltype in  ["gcnn", "cnn"]:
        #     dn = dn * (norm[1]-norm[0]) + norm[0]
        #     dn [dn <= args.threshold] = 0
        return planes2evt(iout, cout)


class DnModel(BaseModel):
    """
    Wrapper class for denoising model.
    """

    def __init__(self, modeltype, ckpt=None):
        """
        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. The path should point to a folder
                    containing a collection and an induction .pth file. If None,
                    an un-trained model will be used.
        """
        super(DnModel, self).__init__(modeltype, "dn", ckpt)


class RoiModel(BaseModel):
    """
    Wrapper class for ROI selection model.
    """

    def __init__(self, modeltype, ckpt=None):
        """
        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        super(RoiModel, self).__init__(modeltype, "roi", ckpt)


class DnRoiModel:
    """
    Wrapper class for denoising and ROI selection model.
    """

    def __init__(self, modeltype, roi_ckpt=None, dn_ckpt=None):
        """
        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        self.roi = RoiModel(modeltype, roi_ckpt)
        self.dn = DnModel(modeltype, dn_ckpt)


def to_dev(*args, dev="cuda:0"):
    """
    Utility class to port list of tensors to device dev. If cuda is not available,
    just passes forward the arguments.

    Parameters
    ----------
        - args: tuple, (induction tensor, collection tensor)
        - dev: str, device to place tensors

    Returns
    -------
        - tuple, (tensor, tensor) ported to cuda:0 device, if cuda is available
    """
    return list(map(lambda x: torch.Tensor(x).to(dev), args))


def print_cfnm(cfnm, channel):
    """
    Prints confusion metrics

    Parameters
    ----------
        - cfnm: list, computed confusion matrix
        - channel: str, available options readout | collection
    """
    tp, fp, fn, tn = cfnm
    logger.info(f"Confusion Matrix on {channel} planes:")
    logger.info(f"\tTrue positives: {tp[0]:.3f} +- {tp[1]:.3f}")
    logger.info(f"\tTrue negatives: {tn[0]:.3f} +- {tn[1]:.3f}")
    logger.info(f"\tFalse positives: {fp[0]:.3f} +- {fp[1]:.3f}")
    logger.info(f"\tFalse negatives: {fn[0]:.3f} +- {fn[1]:.3f}")


def compute_metrics(output, target, task, dev):
    """
    Takes the two events and computes the metrics between their planes,
    separating collection and inductions planes.

    Parameters
    ----------
        - output: np.array, output array of shape=(nb wires, nb tdc ticks)
        - target: np.array, ground truth labels of shape=(nb wires, nb tdc ticks)
        - task: str, available options dn | roi
        - dev: str, computation host device
    """
    if task == "roi":
        metrics = ["bce_dice", "bce", "softdice", "cfnm"]
        helps = ["bce dice", "bce", "softdice", "conf matr"]
    elif task == "dn":
        metrics = ["ssim", "psnr", "mse", "imae"]
        helps = ["stat ssim", "psnr", "mse", "imae"]
    else:
        raise NotImplementedError("Task not implemented")
    metrics_fns = list(map(lambda x: get_loss(x)(reduction="none"), metrics))
    ioutput, coutput = to_dev(*evt2planes(output), dev=dev)
    itarget, ctarget = to_dev(*evt2planes(target), dev=dev)
    iloss = list(map(lambda x: x(ioutput, itarget), metrics_fns))
    closs = list(map(lambda x: x(coutput, ctarget), metrics_fns))

    if task == "roi":
        print_cfnm(iloss[-1], "induction")
        iloss.pop(-1)
        print_cfnm(closs[-1], "collection")
        closs.pop(-1)

    reduce_fn = lambda x: [x.mean(), x.std() / sqrt(len(x))]

    iloss = list(map(reduce_fn, iloss))
    closs = list(map(reduce_fn, closs))

    # loss_ssim computes 1-ssim, print ssim instead
    try:
        idx = metrics.index("ssim")
        iloss[idx][0] = 1 - iloss[idx][0]
        closs[idx][0] = 1 - closs[idx][0]
    except:
        pass

    # loop message function
    msg = "%10s: %.5f +- %.5f"
    loop_fn = lambda x: [msg % (h, l[0], l[1]) for h, l in zip(helps, x)]

    msgs = []

    msgs.append(f"{task_dict[task]}: metrics computation")
    msgs.append("Induction planes:")
    msgs.extend(loop_fn(iloss))
    msgs.append("Collection planes:")
    msgs.extend(loop_fn(closs))

    # log the messages
    logger.info("\n".join(msgs))


# TODO: must fix argument passing in inference
# TODO: must think about saving to output paths
# TODO: check the purpose of to_cuda function
# TODO: think about adding the first two columns in the output array with:
# event number, wire number
