import os
import sys
import argparse
from time import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hitreco import DnRoiModel
from hitreco import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument("--inputs", "-i", type=str, help="The event filename",
                   default="../datasets/20201124/test/evts/p03GeV_cosmics_rawdigit_evt9.npy")
parser.add_argument("--labels", "-l", type=str, help="The event filename",
                   default="../datasets/20201124/test/evts/p03GeV_cosmics_rawdigit_noiseoff_evt9.npy")

def main(inputs, labels):
    threshold = 3.5
    print(f"Denoising event at {inputs}")
    outdir = "./samples/results/"
    evt = np.load(inputs)[:,2:]
    target = np.load(labels)[:,2:]
    model = DnRoiModel("scg")
    dev = "cuda:0"
    dn = model.denoise(evt, dev)
    def save_evt(evtname, ext, evt):
        fname = evtname.split("/")[-1].split(".")[-2].split("_")
        fname.insert(-1,ext)
        fname = outdir + "_".join(fname)
        np.save(fname, evt)
        print(f"Saved output event at {fname}.npy")
    save_evt(inputs, "dn", dn)
    mask = (dn <= threshold) & (dn >= -threshold)
    dn[mask] = 0
    compute_metrics(dn, target, "dn")

if __name__=="__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(f"Program done in {time() - start} s")
