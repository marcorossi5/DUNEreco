import os
import sys
import collections
from math import sqrt
import numpy as np
import torch
from torch.utils.data import DataLoader
from model import SCG_Net
from model_utils import MyDataParallel
from dataloader import InferenceLoader
from train import inference
from losses import get_loss
from args import Args
from analysis.analysis_roi import confusion_matrix
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_yaml

# pdune sp architecture
tdc = 6000 # detector timeticks number
istep = 800 # channel number in induction plane
cstep = 960 # channel number in collection plane
apas = 6
apastep = 2*istep + cstep # number of channels per apa
device_ids = [0]
evstep = apas * apastep # total channel number
ModelTuple = collections.namedtuple('Model', ['induction', 'collection'])
ArgsTuple = collections.namedtuple('Args', ['batch_size', 'patch_stride'])


def evt2planes(event):
    """
    Convert planes to event
    Input:
        event: array-like array
            inputs of shape (evstep, tdc)
            
    Output: np.array
        induction and collection arrays of shape type (N,C,H,W)
    """
    base = np.arange(apas).reshape(-1,1) * apastep
    iidxs = [[0, istep, 2*istep]] + base
    cidxs = [[2*istep, apastep]] + base
    inductions = []
    for start, idx, end in iidxs:
        induction = [event[start:idx], event[idx:end]]
        inductions.extend(induction)
    collections = []
    for start, end in cidxs:
        collections.append(event[start:end])
    return np.stack(inductions)[:,None], np.stack(collections)[:,None]


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
    return planes - medians[:,None,None,None]


def planes2evt(inductions, collections):
    """
    Convert planes to event
    Input:
        inductions, collections: array-like
            inputs of shape type (N,C,H,W)
    Output: np.array
        event array of shape (evstep, tdc)
    """
    inductions = np.array(inductions).reshape(-1,2*istep,tdc)
    collections = np.array(collections)[:,0]
    event = []
    for i, c in zip(inductions, collections):
        event.extend([i, c])
    return np.concatenate(event)

def get_model_and_args(modeltype, model_prefix, task, channel):
    card_prefix = "./denoising/configcards"
    card = f"{modeltype}_{task}_{channel}.yaml"
    parameters = load_yaml(os.path.join(card_prefix, card))
    parameters["channel"] = channel
    args =  Args(**parameters)

    model =  MyDataParallel( SCG_Net(task=args.task, h=args.patch_h,
                                     w=args.w), device_ids=device_ids )
    name = f"{modeltype}_{task}_{channel}.pth"
    fname = os.path.join(model_prefix, name)

    state_dict = torch.load(fname)
    model.load_state_dict(state_dict)
    return ArgsTuple(args.test_batch_size, args.patch_stride), model


def mkModel(modeltype, prefix, task):
    iargs, imodel = get_model_and_args(modeltype, prefix, task, 'induction')
    cargs, cmodel = get_model_and_args(modeltype, prefix, task, 'collection')
    return [iargs, cargs], ModelTuple(imodel, cmodel)


class DnRoiModel:
    def __init__(self, modeltype, prefix='denoising/best_models'):
        """
            Wrapper for inference model
            Parameters:
                modeltype: str
                    "cnn" | "gcnn" | "sgc"
        """
        self.roiargs, self.roi = mkModel(modeltype, prefix, "roi")
        self.dnargs, self.dn = mkModel(modeltype, prefix, "dn")
        self.loader = InferenceLoader    

    def _inference(self, planes, model, args, dev):
        dataset = self.loader(planes)
        loader = DataLoader(dataset=dataset, batch_size=args.batch_size)
        return inference(loader, args.patch_stride, model.to(dev), dev).cpu()

    def roi_selection(self, event, dev):
        """
            Interface for roi selection inference on a complete event
            Parameters:
                event: array-like
                    event input array of shape [wire num, tdcs]
                dev: str
                    "cpu" | "cuda:{n}", device hosting the computation
            Returns:
                np.array
                    event region of interests
        """
        ic = evt2planes(event)
        inductions, collections = map(median_subtraction, ic)
        iout =  self._inference(inductions, self.roi.induction, self.roiargs[0], dev)
        cout =  self._inference(collections, self.roi.collection, self.roiargs[1], dev)
        return planes2evt(iout, cout)        

    def denoise(self, event, dev):
        """
            Interface for roi selection inference on a complete event
            Parameters:
                event: array-like
                    event input array of shape [wire num, tdcs]
            Returns:
                np.array
                    denoised event
        """
        ic = evt2planes(event)
        inductions, collections = map(median_subtraction, ic)
        iout =  self._inference(inductions, self.dn.induction, self.dnargs[0], dev)
        cout =  self._inference(collections, self.dn.collection, self.dnargs[1], dev)
        return planes2evt(iout, cout)


def to_cuda(*args):
    dev = "cuda:0"
    args = list(map(torch.Tensor, args[0]))
    return list(map(lambda x: x.to(dev), args))


def cnfm(output, target):
    # compute the confusion matrix from cuda tensors
    os = output.cpu().numpy()
    ts = target.cpu().numpy()
    n = len(os)
    os = os.reshape([n,-1])
    ts = os.reshape([n,-1])
    cfnm = []
    for o,t in zip(os, ts):
        hit = o[t.astype(int)]
        no_hit = o[1-t.astype(int)]
        cfnm.append( confusion_matrix(hit, no_hit, 0.5) )
    cfnm = np.stack(cfnm)
    
    cfnm = cfnm / cfnm[0,:].sum()
    tp = [cfnm[:,0].mean(), cfnm[:,0].std()/sqrt(n)]
    tn = [cfnm[:,1].mean(), cfnm[:,1].std()/sqrt(n)]
    fp = [cfnm[:,2].mean(), cfnm[:,2].std()/sqrt(n)]
    fn = [cfnm[:,3].mean(), cfnm[:,3].std()/sqrt(n)]

    return [tp, tn, fp, fn]


def print_cfnm(cfnm, channel):
    tp, tn, fp, fn = cfnm
    print(f"Confusion Matrix on {channel} planes:")
    print(f"\tTrue positives: {tp[0]:.3f} +- {tp[1]:.3f}")
    print(f"\tTrue negatives: {tn[0]:.3f} +- {tn[1]:.3f}")
    print(f"\tFalse positives: {fp[0]:.3f} +- {fp[1]:.3f}")
    print(f"\tFalse negatives: {fn[0]:.3f} +- {fn[1]:.3f}")


def compute_metrics(output, target, task):
    """ This function takes the two events and computes the metrics between
    their planes. Separating collection and inductions planes."""
    if task == 'roi':
        metrics = ['bce_dice', 'bce', 'softdice']
    elif task == 'dn':
        metrics = ['ssim', 'psnr', 'mse']
    else:
        raise NotImplementedError("Task not implemented")
    metrics_fns = list(map(lambda x: get_loss(x)(reduction='none'), metrics))
    if task == 'roi':
        metrics_fns.append(cnfm)
    ioutput, coutput = to_cuda(evt2planes(output))
    itarget, ctarget = to_cuda(evt2planes(target))
    iloss = list(map(lambda x: x(ioutput, itarget), metrics_fns))
    closs = list(map(lambda x: x(coutput, ctarget), metrics_fns))
    print(f"Task {task}")
    if task == 'roi':
        print_cfnm(iloss[-1], "induction")
        iloss.pop(-1)
        print_cfnm(closs[-1], "collection")
        closs.pop(-1)
    def reduce(loss):
        sqrtn = sqrt(len(loss))
        return [loss.mean(), loss.std()/sqrtn]
    iloss = list(map(reduce, iloss))
    closs = list(map(reduce, closs))
    print("Induction planes:")
    for metric, loss in zip(metrics, iloss):
        print(f"\t\t {metric:7}: {loss[0]:.5} +- {loss[1]:.5}")
    print("Collection planes:")
    for metric, loss in zip(metrics, closs):
        print(f"\t\t {metric:7}: {loss[0]:.5} +- {loss[1]:.5}")

  
# TODO: must fix argument passing in inference
