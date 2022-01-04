# This file is part of DUNEdn by M. Rossi
import collections
from math import sqrt
import torch
from torch.utils.data import DataLoader
from dunedn.denoising.args import Args
from dunedn.denoising.model import get_model_from_args
from dunedn.denoising.model_utils import Converter, model2batch
from dunedn.denoising.dataloader import InferenceLoader, InferenceCropLoader
from dunedn.denoising.train import inference, identity_inference, gcnn_inference
from dunedn.denoising.losses import get_loss
from dunedn.utils.utils import load_yaml, median_subtraction
from dunedn.configdn import get_dunedn_path
from dunedn.geometry.helpers import evt2planes, planes2evt

ModelTuple = collections.namedtuple("Model", ["induction", "collection"])
ArgsTuple = collections.namedtuple("Args", ["batch_size", "patch_stride", "crop_size"])


def get_model_and_args(modeltype, task, channel, ckpt=None):
    card_prefix = get_dunedn_path()
    card = f"configcards/{modeltype}_{task}_{channel}_configcard.yaml"
    parameters = load_yaml(card_prefix / card)
    parameters["channel"] = channel
    args = Args(**parameters)

    crop_size = None if modeltype == "uscg" else args.crop_size
    patch_stride = args.patch_stride if modeltype == "uscg" else None
    batch_size = model2batch[modeltype][task]

    model = get_model_from_args(args)

    if ckpt is not None:
        fname = ckpt / f"{channel}.pth"
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
    converter = Converter(args.crop_size)
    tiles = converter.planes2tiles(sub_planes)

    dataset = loader(tiles)
    test = DataLoader(dataset=dataset, batch_size=args.batch_size)
    res = gcnn_inference(test, model.to(dev), dev).cpu()
    return converter.tiles2planes(res)


def _identity_inference(planes, loader, **kwargs):
    dataset = loader(planes)
    test = DataLoader(dataset=dataset)
    return identity_inference(test).cpu()


def get_inference(modeltype, **kwargs):
    if modeltype == "uscg":
        return _scg_inference(**kwargs)
    elif modeltype in ["cnn", "gcnn"]:
        return _gcnn_inference(**kwargs)
    elif modeltype == "id":
        return _identity_inference(**kwargs)


class BaseModel:
    def __init__(self, modeltype, task, ckpt=None):
        """
        Wrapper for base model.

        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "usgc"
            - task: str, valid options "dn" | "roi"
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
    def __init__(self, modeltype, ckpt=None):
        """
        Wrapper for denoising model.

        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. The path should point to a folder
                    containing a collection and an induction .pth file. If None,
                    an un-trained model will be used.
        """
        super(DnModel, self).__init__(modeltype, "dn", ckpt)


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
        super(RoiModel, self).__init__(modeltype, "roi", ckpt)


class DnRoiModel:
    def __init__(self, modeltype, roi_ckpt=None, dn_ckpt=None):
        """
        Wrapper for inference model.

        Parameters
        ----------
            - modeltype: str, valid options: "cnn" | "gcnn" | "sgc"
            - ckpt: Path, saved checkpoint path. If None, an un-trained model
                    will be used
        """
        self.roi = RoiModel(modeltype, roi_ckpt)
        self.dn = DnModel(modeltype, dn_ckpt)


def to_cuda(*args):
    if not torch.cuda.is_available():
        return args
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
# TODO: check the purpose of to_cuda function
# TODO: think about adding the first two columns in the output array with:
# event number, wire number
