# This file is part of DUNEdn by M. Rossi
import os
import argparse
from time import time as tm
import numpy as np
import torch
from skimage.feature import canny

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir_name",
    "-p",
    default="../datasets",
    type=str,
    help="Directory path to datasets",
)


def draw_results(a, b, c, d):
    tot = a + b + c + d
    print("Over a total of %d pixels:\n" % tot)
    print("------------------------------------------------")
    print("|{:>20}|{:>12}|{:>12}|".format("", "Signal", "Background"))
    print("------------------------------------------------")
    print("|{:>20}|{:>12.4e}|{:>12.4e}|".format("Predicted signal", a / tot, b / tot))
    print("------------------------------------------------")
    print(
        "|{:>20}|{:>12.4e}|{:>12.4e}|".format("Predicted background", c / tot, d / tot)
    )
    print("------------------------------------------------")
    print("{:>21}|{:>12}|{:>12}|".format("", "Sensitivity", "Specificity"))
    print("                     ---------------------------")
    print("{:>21}|{:>12.4e}|{:>12.4e}|".format("", a / (a + c), d / (b + d)))
    print("                     ---------------------------\n")


def main(dir_name):

    for s in ["readout_", "collection_"]:
        for ss in ["train", "val", "test"]:
            clear = np.array(
                torch.load(os.path.join(dir_name, "clear_planes", "".join([s, ss])))
            )

            edges = []
            for c in clear:
                edges.append(canny(np.array(c)).astype(float))

            edges = np.stack(edges, 0)
            clear[clear != 0] = 1

            d = clear * 10 - edges

            tp = (d[d == 9].shape)[0]
            tn = (d[d == 0].shape)[0]
            fn = (d[d == 10].shape)[0]
            fp = (d[d == -1].shape)[0]

            print("".join(["\n", s, ss]))
            draw_results(tp, fp, fn, tn)


if __name__ == "__main__":
    args = vars(parser.parse_args())
    start = tm()
    main(**args)
    print("\nProgram done in %f" % (tm() - start))
