import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hitreco import DnRoiModel


if __name__=="__main__":
    evt = np.load("../datasets/denoising/test/evts/out_noiseon_evt9")[:,2:]
    #evt = np.load("../datasets/20201124/0.3GeV/p03GeV_rawdigits_evt1.npy")[:,2:]
    model = DnRoiModel("cnn")
    dev = "cuda:0"
    roi = model.roi_selection(evt, dev)
    dn = model.denoise(evt, dev)
    np.save("samples/roi_evt", roi)
    np.save("samples/dn_evt", dn)
