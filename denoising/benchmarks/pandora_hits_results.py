"""
This module computes recob::hits results for planes in the test set

Threshold is the cut below which a clear raw digit is considered an
hit with signal.

Output file contains accuracy, sensitivity and specificity provided
with mean values and uncertainties.
"""
import os
import sys
import argparse
import numpy as np
import time as tm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis.analysis_roi import confusion_matrix


PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--dirname",
    "-p",
    default="../datasets/backup/test",
    type=str,
    help="Directory path to datasets",
)
PARSER.add_argument(
    "--threshold",
    "-t",
    default=3.5,
    type=float,
    help="Threshold to distinguish signal/noise in labels",
)


def main(dirname, threshold):
    # Load true hits
    # TODO: must subtract pedestal!
    file_name = os.path.join(dirname, "planes", "collection_clear.npy")
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
    file_name = os.path.join(dirname, "benchmark/hits", "pandora_collection_hits.npy")

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
    args = vars(PARSER.parse_args())
    START = tm.time()
    main(**args)
    print("Program done in %f" % (tm.time() - START))
