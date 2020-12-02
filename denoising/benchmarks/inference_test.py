import os
import sys
import argparse
from time import time
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hitreco import DnRoiModel

parser = argparse.ArgumentParser()
parser.add_argument("--evtname", "-e", type=str, help="The event filename",
                   default="../datasets/20201124/0.3GeV/p03GeV_rawdigits_evt1.npy")

def main(evtname):
    print(f"Denoising event at {evtname}")
    evt = np.load(evtname)[:,2:]
    model = DnRoiModel("scg")
    dev = "cuda:0"
    dn = model.denoise(evt, dev)
    def save_evt(fname, evt):
        np.save(fname, evt)
        print(f"Saved output event at {fname}.npy")
    save_evt('samples/dn_event', dn)

if __name__=="__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(f"Program done in {time() - start} s")
