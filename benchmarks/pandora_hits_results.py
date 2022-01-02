"""
This module computes recob::hits results for planes in the test set

Threshold is the cut below which a clear raw digit is considered an
hit with signal.

Output file contains accuracy, sensitivity and specificity provided
with mean values and uncertainties.
"""
import argparse
import numpy as np
from time import time as tm
from pathlib import Path
from dunedn.utils.utils import confusion_matrix


def main(dirname, threshold):
    # Load true hits
    # TODO: must subtract pedestal!
    file_name = dirname / "planes/collection_clear.npy"
    true_hits = np.load(file_name)
    mask = np.logical_and(true_hits >= 0, true_hits <= threshold)
    true_hits[mask] = 0
    true_hits[~mask] = 1

    sig = np.count_nonzero(true_hits)
    tot = true_hits.size

    print("Percentage of pixels with signal:", sig / tot)
    true_hits = true_hits.astype(bool)

    # TODO: calculate acc, sns, spc (mean +- inc)

    # Load recob::hits
    file_name = dirname / "benchmark/hits/pandora_collection_hits.npy"
    pandora_hits = np.load(file_name)

    acc = []
    sns = []
    spc = []

    for pandora_hit, true_hit in zip(pandora_hits, true_hits):
        hit = pandora_hit[true_hit]
        no_hit = pandora_hit[~true_hit]

        tp, fp, fn, tn = confusion_matrix(hit, no_hit)
        acc.append((tp + tn) / (tp + fp + fn + tn))
        sns.append(tp / (fn + tp))
        spc.append(tn / (fp + tn))

    acc_mean = np.mean(acc)
    acc_std = np.std(acc) / np.sqrt(len(acc))

    sns_mean = np.mean(sns)
    sns_std = np.std(sns) / np.sqrt(len(sns))

    spc_mean = np.mean(spc)
    spc_std = np.std(spc) / np.sqrt(len(spc))

    res = np.array([[acc_mean, acc_std], [sns_mean, sns_std], [spc_mean, spc_std]])

    print(res)

    dir_name = "denoising/benchmarks/results/"
    fname = dir_name + "pandora_hits_metrics"
    np.save(fname, res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dirname",
        default="../datasets/backup/test",
        type=Path,
        help="Directory path to datasets",
    )
    parser.add_argument(
        "--threshold",
        default=3.5,
        type=float,
        help="Threshold to distinguish signal/noise in labels",
    )
    args = vars(parser.parse_args())
    start = tm()
    main(**args)
    print("Program done in %f" % (tm()-start))
