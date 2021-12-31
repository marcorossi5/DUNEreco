import os
import collections
from math import sqrt
import numpy as np
import torch
from torch.utils.data import DataLoader
from dunedn.denoising.model import get_model
from dunedn.denoising.model_utils import MyDataParallel
from dunedn.denoising.model_utils import Converter
from dunedn.denoising.dataloader import InferenceLoader, InferenceCropLoader
from dunedn.denoising.train import inference, identity_inference, gcnn_inference
from dunedn.denoising.losses import get_loss
from dunedn.denoising.args import Args
from dunedn.utils.utils import load_yaml

# pdune sp architecture
tdc = 6000  # detector timeticks number
istep = 800  # channel number in induction plane
cstep = 960  # channel number in collection plane
apas = 6
apastep = 2 * istep + cstep  # number of channels per apa
evstep = apas * apastep  # total channel number
ModelTuple = collections.namedtuple("Model", ["induction", "collection"])
ArgsTuple = collections.namedtuple("Args", ["batch_size", "patch_stride", "patch_size"])


model2batch = {
    "scg": {"dn": 1, "roi": 1},
    "gcnn": {"dn": 128, "roi": 512},
    "cnn": {"dn": 376, "roi": 2048},
}


def evt2planes(event):
    """
    Convert planes to event
    Input:
        event: array-like array
            inputs of shape (evstep, tdc)

    Output: np.array
        induction and collection arrays of shape type (N,C,H,W)
    """
    base = np.arange(apas).reshape(-1, 1) * apastep
    iidxs = [[0, istep, 2 * istep]] + base
    cidxs = [[2 * istep, apastep]] + base
    inductions = []
    for start, idx, end in iidxs:
        induction = [event[start:idx], event[idx:end]]
        inductions.extend(induction)
    collections = []
    for start, end in cidxs:
        collections.append(event[start:end])
    return np.stack(inductions)[:, None], np.stack(collections)[:, None]


def median_subtraction(planes):
    """
    Subtract median value from input planes
    Input:
        planes: np.array
            array of shape (N,C,H,W)
    Output: np.array
        median subtracted planes ( =dim(N,C,H,W))
    """
    shape = [planes.shape[0], -1]
    medians = np.median(planes.reshape(shape), axis=1)
    return planes - medians[:, None, None, None]


def planes2evt(inductions, collections):
    """
    Convert planes to event
    Input:
        inductions, collections: array-like
            inputs of shape type (N,C,H,W)
    Output: np.array
        event array of shape (evstep, tdc)
    """
    inductions = np.array(inductions).reshape(-1, 2 * istep, tdc)
    collections = np.array(collections)[:, 0]
    event = []
    for i, c in zip(inductions, collections):
        event.extend([i, c])
    return np.concatenate(event)


def get_model_and_args(modeltype, model_prefix, task, channel):
    card_prefix = "./denoising/configcards"
    card = f"{modeltype}_{task}_{channel}.yaml"
    parameters = load_yaml(os.path.join(card_prefix, card))
    parameters["channel"] = channel
    args = Args(**parameters)

    patch_size = "None" if modeltype == "scg" else eval(args.patch_size)
    patch_stride = args.patch_stride if modeltype == "scg" else None
    batch_size = model2batch[modeltype][task]

    # TODO: when changing the models inputs, this has to be changed accordingly
    kwargs = {}
    if modeltype == "scg":
        kwargs["task"] = args.task
        kwargs["h"] = args.patch_h
        kwargs["w"] = args.w
        device_ids = [0]
    elif modeltype in ["cnn", "gcnn"]:
        kwargs["model"] = modeltype
        kwargs["task"] = task
        kwargs["channel"] = channel
        kwargs["patch_size"] = patch_size
        kwargs["input_channels"] = args.input_channels
        kwargs["hidden_channels"] = args.hidden_channels
        kwargs["k"] = args.k
        kwargs["dataset_dir"] = args.dataset_dir
        kwargs["normalization"] = args.normalization
        device_ids = [0, 1, 2, 3]
    else:
        raise NotImplementedError("Loss function not implemented")

    model = MyDataParallel(get_model(modeltype, **kwargs), device_ids=device_ids)
    name = f"{modeltype}_{task}_{channel}.pth"
    fname = os.path.join(model_prefix, name)

    state_dict = torch.load(fname)
    model.load_state_dict(state_dict)
    return ArgsTuple(batch_size, patch_stride, patch_size), model


def mkModel(modeltype, prefix, task):
    if modeltype == "id":
        return None, None
    iargs, imodel = get_model_and_args(modeltype, prefix, task, "induction")
    cargs, cmodel = get_model_and_args(modeltype, prefix, task, "collection")
    return [iargs, cargs], ModelTuple(imodel, cmodel)


def _scg_inference(planes, loader, model, args, dev):
    dataset = loader(planes)
    test = DataLoader(dataset=dataset, batch_size=args.batch_size)
    return inference(test, args.patch_stride, model.to(dev), dev).cpu()


def _gcnn_inference(planes, loader, model, args, dev):
    # creating a new instance of converter every time could waste time if the
    # inference is called many times.
    # TODO: think about to make it a DnRoiModel attribute and pass it to the fn
    # TODO: the batch size changes according to task, modeltype
    sub_planes = torch.Tensor(median_subtraction(planes))
    converter = Converter(args.patch_size)
    tiles = converter.planes2tiles(sub_planes)

    dataset = loader(tiles)
    test = DataLoader(dataset=dataset, batch_size=args.batch_size)
    res = gcnn_inference(test, model.to(dev), dev).cpu()
    return converter.tiles2planes(res)


def _identity_inference(planes, loader, args, dev):
    dataset = loader(planes)
    test = DataLoader(dataset=dataset, batch_size=args.batch_size)
    return identity_inference(test).cpu()


def get_inference(modeltype, **kwargs):
    if modeltype == "scg":
        return _scg_inference(**kwargs)
    elif modeltype in ["cnn", "gcnn"]:
        return _gcnn_inference(**kwargs)
    elif modeltype == "id":
        return _identity_inference(**kwargs)


class BaseModel:
    def __init__(self, modeltype, ckpt=None):
        """
        Wrapper for base model.

        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        self.modeltype = modeltype
        self.loader = InferenceLoader if modeltype == "scg" else InferenceCropLoader

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
    def __init__(self, modeltype, ckpt=None):
        """
        Wrapper for denoising model.

        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        super(DnModel, self).__init__(modeltype, ckpt)
        self.args, self.model = mkModel(modeltype, ckpt, "dn")


class RoiModel(BaseModel):
    def __init__(self, modeltype, ckpt=None):
        """
        Wrapper for ROI selection model.

        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        super(RoiModel, self).__init__(modeltype, ckpt)
        self.args, self.model = mkModel(modeltype, ckpt, "dn")


class DnRoiModel:
    def __init__(self, modeltype, ckpt="denoising/best_models"):
        """
        Wrapper for inference model.

        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        self.dn = DnModel(modeltype, ckpt)
        self.roi = RoiModel(modeltype, ckpt)


def to_cuda(*args):
    dev = "cuda:0"
    args = list(map(torch.Tensor, args[0]))
    return list(map(lambda x: x.to(dev), args))


def print_cfnm(cfnm, channel):
    tp, fp, fn, tn = cfnm
    print(f"Confusion Matrix on {channel} planes:")
    print(f"\tTrue positives: {tp[0]:.3f} +- {tp[1]:.3f}")
    print(f"\tTrue negatives: {tn[0]:.3f} +- {tn[1]:.3f}")
    print(f"\tFalse positives: {fp[0]:.3f} +- {fp[1]:.3f}")
    print(f"\tFalse negatives: {fn[0]:.3f} +- {fn[1]:.3f}")


def compute_metrics(output, target, task):
    """This function takes the two events and computes the metrics between
    their planes. Separating collection and inductions planes."""
    if task == "roi":
        metrics = ["bce_dice", "bce", "softdice", "cfnm"]
    elif task == "dn":
        metrics = ["ssim", "psnr", "mse", "imae"]
    else:
        raise NotImplementedError("Task not implemented")
    metrics_fns = list(map(lambda x: get_loss(x)(reduction="none"), metrics))
    ioutput, coutput = to_cuda(evt2planes(output))
    itarget, ctarget = to_cuda(evt2planes(target))
    iloss = list(map(lambda x: x(ioutput, itarget), metrics_fns))
    closs = list(map(lambda x: x(coutput, ctarget), metrics_fns))
    print(f"Task {task}")
    if task == "roi":
        print_cfnm(iloss[-1], "induction")
        iloss.pop(-1)
        print_cfnm(closs[-1], "collection")
        closs.pop(-1)

    def reduce(loss):
        sqrtn = sqrt(len(loss))
        return [loss.mean(), loss.std() / sqrtn]

    iloss = list(map(reduce, iloss))
    closs = list(map(reduce, closs))
    print("Induction planes:")
    for metric, loss in zip(metrics, iloss):
        print(f"\t\t loss {metric:7}: {loss[0]:.5} +- {loss[1]:.5}")
    print("Collection planes:")
    for metric, loss in zip(metrics, closs):
        print(f"\t\t loss {metric:7}: {loss[0]:.5} +- {loss[1]:.5}")


# TODO: must fix argument passing in inference
# TODO: must think about saving to output paths
