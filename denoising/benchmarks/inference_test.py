import os
import sys
import nummpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hitreco import DnRoiModel


if __name__=="__main__":
    evt = np.load("../datasets/denoising/test/evts/out_noiseon_evt9")
    model = DnRoiModel("cnn")
    roi = model.roi_selection(evt, "cuda:0")
    dn = model.denoise(evt, "cuda:0")