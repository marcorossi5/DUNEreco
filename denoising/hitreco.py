import os
import sys
import collections
import numpy as np
from torch.utils.data import DataLoader
from model import DenoisingModel
from dataloader import PlaneLoader
from train import inference
from args import Args
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import load_yaml

# pdune sp architecture
tdc = 6000 # detector timeticks number
istep = 800 # channel number in induction plane
cstep = 960 # channel number in collection plane
apas = 6
apastep = 2*istep + cstep # number of channels per apa
evstep = apas * apastep # total channel number
ModelTuple = collections.namedtuple('Model', ['induction', 'collection'])
ArgsTuple = collections.namedtuple('Args', ['patch_size', 'batch_size'])


def evt2planes(event):
    """
    Convert planes to event
    Input:
        event: array-like array
            inputs of shape (evstep, tdc)
            
    Output: np.array
        induction and collection arrays of shape type (N,C,H,W)
    """
    event = event[None]
    base = np.arange(apas).reshape(-1,1) * apastep
    iidxs = [[0, istep, 2*istep]] + base
    cidxs = [[2*istep, apastep]] + base
    inductions = []
    for start, idx, end in iidxs:
        induction = [event[:,start:idx], event[:,idx:end]]
        inductions.extend(induction)
    collections = []
    for start, end in cidxs:
        collections.append(event[:,start:end])
    return np.stack(inductions), np.stack(collections)

    
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

    model =  DenoisingModel(args)
    prefix = "denoising/best_models"
    name = f"{modeltype}_{task}"
    fname = os.path.join(prefix, name + f"_{channel}.dat" )
    model.load_state_dict(torch.load(fname))
    return ArgsTuple(args.patch_size, args.test_batch_size), model


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
                    "cnn" | "gcnn"
        """
        self.roiargs, self.roi = mkModel(modeltype, "roi")
        self.dnargs, self.dn = mkModel(modeltype, "dn")
        self.loader = PlaneLoader    

    def _inference(self, event, models, args, dev):
        inductions, collections = evt2planes(event[None])
        iset = self.loader(args[0], planes=inductions)
        cset = self.loader(args[1], planes=collections)

        iloader = DataLoader(dataset=iset, batch_size=args[0].batch_size)
        cloader = DataLoader(dataset=cset, batch_size=args[1].batch_size)

        iout = inference(iloader, models.induction, dev)
        cout = inference(cloader, models.collection, dev)
        return planes2evt(iout, cout)


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
        return self._inference(event, self.roi.to(dev), self.roiargs, dev)
        

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
        return self._inference(event, self.dn.to(dev), self.dnargs, dev)
    
