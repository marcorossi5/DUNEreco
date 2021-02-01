import numpy as np
import argparse
from time import time
from hitreco import evt2planes

parser = argparse.ArgumentParser()
parser.add_argument("--fname", "-f",  type=str, help="Event root folder",
                   default="../datasets/20201124/test/evts/p2GeV_cosmics_wire_ev9.npy")


def fit_constant(wires, clear):
    from ROOT import TGraph, TF1

    ratios = clear.sum(-1)/wires.sum(-1)
    masks = wires.sum(-1) != 0

    fits = []
    for ratio, mask in zip(ratios, masks):
        points = ratio[mask]
        shape = points.shape[0]
        g = TGraph(shape, np.arange(shape).astype(float), points)
        func = TF1('func', '[0]', 0, 10)
        fit = g.Fit('func', 'SQC').Get()
        fits.append([fit.Parameter(0), fit.Error(0)])

    return np.array(fits)


def compute_constants(wires, labels):
    k = fit_constant(wires, labels)
    print(k)
    return k


def main(fname):
    wires = np.load(fname)
    _, cwires = evt2planes(wires)

    fname = fname.replace("wire", "rawdigit_noiseoff")
    labels = np.load(fname)
    _, clabels = evt2planes(labels)

    # k = compute_constants(cwires, clabels)
    k = np.array([[3.55240744, 0.09703621],
                  [1.75292430, 0.02975895],
                  [1.94628324, 0.03519356],
                  [1.68464956, 0.02822732],
                  [1.36381451, 0.02297976],
                  [1.69213512, 0.02634354]])
    
    from hitreco import compute_metrics
    compute_metrics(cwires*k[:,0], clabels, 'dn')



if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = time()
    main(**args)
    print(f"Program done in {time() - start}")