import os
import sys
import argparse
from time import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hitreco import DnRoiModel
from hitreco import compute_metrics

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    "-r",
    type=str,
    help="Event root folder",
    default="../datasets/20201124/test/evts",
)
parser.add_argument(
    "--energy", "-e", type=str, help="Proton beam energy (no separator .)", default="03"
)
parser.add_argument(
    "--modeltype", "-m", type=str, help="Model type scg | gcnn | cnn", default="scg"
)
parser.add_argument(
    "--prefix",
    type=str,
    help="Folder for stored best models",
    default="denoising/best_models",
)
parser.add_argument(
    "--suffix", type=str, help="File to process, overrides energy", default=None
)
parser.add_argument(
    "--old",
    action="store_true",
    help="Use sim::SimChannel, e- to ADC counts conversion",
)

outdir = "./samples/results/"


def main(root_path, modeltype, energy, prefix, suffix, old):
    def save_evt(evtname, ext, evt):
        fname = evtname.split("/")[-1].split(".")[-2].split("_")
        fname.insert(-1, ext)
        fname = outdir + "_".join(fname)
        np.save(fname, evt)
        print(f"Saved output event at {fname}.npy")

    if suffix is None:
        file_name = f"p{energy}GeV_cosmics_rawdigit_evt9.npy"
        fname = os.path.join(root_path, file_name)
        ext = "rawdigit_noiseoff" if old else "simch_labels"
        label_fname = fname.replace("rawdigit", ext)
    else:
        fname = os.path.join(root_path, suffix)
        label_fname = fname.replace("noiseon", "noiseoff")

    ElectronsToADC = 6.8906513e-3
    if not old:
        print("Converting number of electrons to ADC counts")
    mult = 1 if old else ElectronsToADC
    threshold = 3.5

    print(f"Denoising event at {fname}")
    evt = np.load(fname)[:, 2:]
    target = np.load(label_fname)[:, 2:] * mult
    model = DnRoiModel(modeltype, prefix=prefix)
    dev = "cuda:0"

    dn = model.denoise(evt, dev)
    save_evt(fname, "dn", dn)
    mask = (dn <= threshold) & (dn >= -threshold)
    dn[mask] = 0
    compute_metrics(dn, target, "dn")

    roi = model.roi_selection(evt, dev)

    mask = (target <= threshold) & (target >= -threshold)
    target[mask] = 0
    target[~mask] = 1
    compute_metrics(roi, target, "roi")
    save_evt(fname, "roi", roi)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(f"Program done in {time() - start} s")
