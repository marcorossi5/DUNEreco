# This file is part of DUNEdn by M. Rossi
import numpy as np
import argparse
from time import time as tm
from dunedn.denoising.hitreco import evt2planes, planes2evt

parser = argparse.ArgumentParser()
parser.add_argument(
    "--fname",
    "-f",
    type=str,
    help="Event root folder",
    default="../datasets/20201124/test/evts/p2GeV_cosmics_wire_evt9.npy",
)
parser.add_argument(
    "--fit", action="store_true", help="Use PyROOT to fit the normalization constant"
)

dev = "cuda:0"


def fit_constant(wires, clear):
    from ROOT import TGraph, TF1

    ratios = clear.sum(-1) / wires.sum(-1)
    masks = wires.sum(-1) != 0

    fits = []
    for ratio, mask in zip(ratios, masks):
        points = ratio[mask]
        shape = points.shape[0]
        g = TGraph(shape, np.arange(shape).astype(float), points)
        func = TF1("func", "[0]", 0, 10)
        fit = g.Fit("func", "SQC").Get()
        fits.append([fit.Parameter(0), fit.Error(0)])

    return np.array(fits)


def compute_constants(wires, labels):
    k = fit_constant(wires, labels)
    print(k)
    return k


def main(fname, fit):
    wires = np.load(fname)[:, 2:]
    iwires, cwires = evt2planes(wires)

    fname = fname.replace("wire", "rawdigit_noiseoff")
    labels = np.load(fname)[:, 2:]
    ilabels, clabels = evt2planes(labels)

    if fit:
        ik = compute_constants(iwires, ilabels)
        ck = compute_constants(cwires, clabels)
    else:
        ik = np.array(
            [
                [6.44518276e-01, 6.31853618e-01],
                [1.24100913e00, 1.59996499e-02],
                [1.01106692e00, 1.58423364e-02],
                [1.03538924e00, 6.87743815e-03],
                [7.61434061e-01, 1.39250640e-02],
                [9.94644122e-01, 5.43702747e-03],
                [7.39669829e-01, 8.73465559e-03],
                [9.87428925e-01, 5.24721814e-03],
                [1.01346316e00, 6.88380304e-03],
                [1.02172429e00, 1.78829187e-03],
                [1.02850125e00, 5.42467795e-03],
                [1.02023728e00, 1.01015753e-03],
            ]
        )
        ck = np.array(
            [
                [3.35929051, 0.09637187],
                [2.07739054, 0.04431658],
                [1.74806175, 0.02859704],
                [1.83722203, 0.03261106],
                [1.37893158, 0.02219072],
                [1.32170305, 0.01274681],
            ]
        )
    inormalized = iwires * ik[:, :1, None, None]
    cnormalized = cwires * ck[:, :1, None, None]
    normalized = planes2evt(inormalized, cnormalized)

    from hitreco import compute_metrics

    compute_metrics(normalized, labels, "dn")


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = tm()
    main(**args)
    print(f"Program done in {tm() - start}")
