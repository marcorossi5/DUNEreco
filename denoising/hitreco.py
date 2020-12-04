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

def get_model_and_args(modeltype, task, channel):
    prefix = "./denoising/configcards"
    card = f"{modeltype}_{task}.yaml"
    parameters = load_yaml(os.path.join(prefix, card))
    parameters["channel"] = channel
    args =  Args(**parameters)

    model =  MyDataParallel( SCG_Net(task=args.task, h=args.patch_h,
                                     w=args.w), device_ids=device_ids )
    prefix = "denoising/best_models"
    name = f"{modeltype}_{task}_{channel}.pth"
    fname = os.path.join(prefix, name)

    state_dict = torch.load(fname)
    model.load_state_dict(state_dict)
    return ArgsTuple(args.test_batch_size, args.patch_stride), model


def mkModel(modeltype, task):
    iargs, imodel = get_model_and_args(modeltype, task, 'induction')
    cargs, cmodel = get_model_and_args(modeltype, task, 'collection')
    return [iargs, cargs], ModelTuple(imodel, cmodel)


class DnRoiModel:
    def __init__(self, modeltype):
        """
            Wrapper for inference model
            Parameters:
                modeltype: str
                    "cnn" | "gcnn" | "sgc"
        """
        self.roiargs, self.roi = mkModel(modeltype, "roi")
        self.dnargs, self.dn = mkModel(modeltype, "dn")
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
        inductions, collections = evt2planes(event)
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
        inductions, collections = evt2planes(event)
        iout =  self._inference(inductions, self.dn.induction, self.dnargs[0], dev)
        cout =  self._inference(collections, self.dn.collection, self.dnargs[1], dev)
        return planes2evt(iout, cout)


def to_cuda(*args):
    dev = "cuda:0"
    args = list(map(torch.Tensor, args[0]))
    return list(map(lambda x: x.to(dev), args))


def compute_metrics(output, target, task):
    """ This function takes the two events and computes the metrics between
    their planes. Separating collection and inductions planes."""
    if task == 'roi':
        metrics = ['bce', 'softdice']
    elif task == 'dn':
        metrics = ['ssim', 'psnr', 'mse']
    else:
        raise NotImplementedError("Task not implemented")
    metrics_fns = list(map(lambda x: get_loss(x)(reduction='none'), metrics))
    ioutput, coutput = to_cuda(evt2planes(output))
    itarget, ctarget = to_cuda(evt2planes(target))
    iloss = list(map(lambda x: x(ioutput, itarget), metrics_fns))
    closs = list(map(lambda x: x(coutput, ctarget), metrics_fns))
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
