""" This module returns the ROIs from recob::hit objects"""
import os
import sys
import argparse
import numpy as np
import glob
import time as tm

parser = argparse.ArgumentParser()
parser.add_argument("--dirname", "-p", default="../datasets/IML2020/test/benchmark",
                    type=str, help='Directory path to datasets')

N_CHANNELS = 2560
N_INDUCTION = 800
N_COLLECTION = 960
N_APAS = 6
N_TICKS = 6000


def process_hits(fname):
    hits = np.load(os.path.join(fname))

    ROIs = np.zeros((N_APAS, N_COLLECTION, N_TICKS))

    for apa in range(N_APAS):
        first_ch = N_CHANNELS*apa + 2*N_INDUCTION
        last_ch = N_CHANNELS*(apa+1)
        mask = np.logical_and(hits[:, 0] >= first_ch, hits[:, 0] < last_ch)

        for hit in hits[mask]:
            ch = hit[0] - first_ch
            ROIs[apa, ch, hit[1]:hit[2]] = 1
    return ROIs


def process_wires(fname):
    wires = np.load(os.path.join(fname))

    coll_wires = []
    for apa in range(N_APAS):
        first_ch = N_CHANNELS*apa + 2*N_INDUCTION
        last_ch = N_CHANNELS*(apa+1)
        coll_wires.append(wires[first_ch:last_ch, 2:])
    return coll_wires


def process_hits_and_dump(dirname):
    """
    Processes hits to cast them into an array of shape (N_CHANNELS, N_TICKS).
    Saves an array named collection_hits in benchmark/hits folder with all 
    collection region of interests in the datasetto be used as a benchmark.
    Shape: (ALL_APAS, N_COLLECTION, N_TICKS)
    """
    input_dir = os.path.join(dirname, "pandora_out")
    output_dir = os.path.join(dirname, "hits")
    fnames = glob.glob(os.path.join(input_dir, "*recobhits*"))

    ROIs = []
    for fname in fnames:
        ROIs.append(process_hits(fname))

    ROIs = np.concatenate(ROIs)
    outname = os.path.join(output_dir, "pandora_collection_hits")
    np.save(outname, ROIs)


def process_wires_and_dump(dirname):
    """
    Saves all collection planes array into benchmark/wires folder.
    Saves arrays named ..._collection.npy into benchmark/pandora_out folder.
    Shape: (ALL_APAS, N_COLLECTION, N_TICKS)

    Note: the ADC counts are normalized arbitrarily
    """
    input_dir = os.path.join(dirname, "pandora_out")
    output_dir = os.path.join(dirname, "wires")
    fnames = glob.glob(os.path.join(input_dir, "*recobwires*"))

    wires = []
    for fname in fnames:
        wires.extend(process_wires(fname))
    wires = np.stack(wires)
    outname = os.path.join(output_dir, "pandora_collection_wires")
    np.save(outname, wires)


def main(dirname):
    process_hits_and_dump(dirname)

    process_wires_and_dump(dirname)


if __name__ == '__main__':
    args = vars(parser.parse_args())
    START = tm.time()
    main(**args)
    print('Program done in %f' % (tm.time()-START))
