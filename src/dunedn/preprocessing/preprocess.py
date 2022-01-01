# This file is part of DUNEdn by M. Rossi
import os, glob
import glob
import numpy as np
from dunedn.preprocessing.putils import get_crop


def add_arguments_preprocess(parser):
    parser.add_argument(
        "-p",
        default="../datasets/denoising",
        type=str,
        help="directory path to datasets",
        metavar="DIR_NAME",
    )
    parser.add_argument(
        "-n",
        default=5000,
        type=int,
        help="number of crops for each plane",
        metavar="N_CROPS",
    )
    parser.add_argument(
        "-c", default=32, type=int, help="crop edge", metavar="CROP_EDGE"
    )
    parser.add_argument(
        "-x", default=0.5, type=float, help="percentage of signal", metavar="PERCENTAGE"
    )
    parser.set_defaults(func=preprocess)


def preprocess(args):
    # TODO: remove this step and pass NameSpace object
    args = vars(args)
    args.pop("func")
    print("Args:")
    for k in args.keys():
        print(k, args[k])
    preprocess_main(**args)


# TODO: remove this global variables, write a config.py and a geometry_helper.py
# files. The geometry_helper module should contain the functions for converting
# events array into planes and back.

# patch_size = (32,32)
apas = 6
istep = 800  # induction plane channel number
cstep = 960  # collection plane channel number
apastep = 2 * istep + cstep


def get_planes_and_dump(dname):
    base = np.arange(apas).reshape(-1, 1) * apastep
    iidxs = [[0, istep, 2 * istep]] + base
    cidxs = [[2 * istep, apastep]] + base

    iclear = []
    inoisy = []
    isimch = []
    cclear = []
    cnoisy = []
    csimch = []

    path_clear = glob.glob(os.path.join(dname, "evts/*noiseoff*"))

    assert len(path_clear) != 0

    for file_clear in path_clear:
        file_noisy = file_clear.replace("rawdigit_noiseoff", "rawdigit")
        file_simch = file_clear.replace("rawdigit_noiseoff", "simch_labels")
        print(f"Processing files:\n\t{file_clear}\n\t{file_noisy}\n\t{file_simch}")
        c = np.load(file_clear)[:, 2:]
        n = np.load(file_noisy)[:, 2:]
        s = np.load(file_simch)[:, 2:]
        for start, idx, end in iidxs:
            iclear.extend([c[start:idx], c[idx:end]])
            inoisy.extend([n[start:idx], n[idx:end]])
            isimch.extend([s[start:idx], s[idx:end]])
        for start, end in cidxs:
            cclear.append(c[start:end])
            cnoisy.append(n[start:end])
            csimch.append(s[start:end])

    cclear = np.stack(cclear, 0)[:, None]
    cnoisy = np.stack(cnoisy, 0)[:, None]
    csimch = np.stack(csimch, 0)[:, None]
    iclear = np.stack(iclear, 0)[:, None]
    inoisy = np.stack(inoisy, 0)[:, None]
    isimch = np.stack(isimch, 0)[:, None]

    print("\tCollection clear planes: ", cclear.shape)
    print("\tCollection noisy planes: ", cnoisy.shape)
    print("\tCollection sim::SimChannel planes: ", csimch.shape)
    print("\tInduction clear planes: ", iclear.shape)
    print("\tInduction noisy planes: ", inoisy.shape)
    print("\tInduction sim::SimChannel planes: ", isimch.shape)

    fname = os.path.join(dname, f"planes/induction_clear")
    np.save(fname, np.stack(iclear, 0))

    fname = os.path.join(dname, f"planes/induction_noisy")
    np.save(fname, np.stack(inoisy, 0))

    fname = os.path.join(dname, f"planes/induction_simch")
    np.save(fname, np.stack(isimch, 0))

    fname = os.path.join(dname, f"planes/collection_clear")
    np.save(fname, np.stack(cclear, 0))

    fname = os.path.join(dname, f"planes/collection_noisy")
    np.save(fname, np.stack(cnoisy, 0))

    fname = os.path.join(dname, f"planes/collection_simch")
    np.save(fname, np.stack(csimch, 0))

    fname = os.path.join(dname, f"planes/sample_collection_clear")
    np.save(fname, np.stack(cclear[:10], 0))

    fname = os.path.join(dname, f"planes/sample_collection_noisy")
    np.save(fname, np.stack(cnoisy[:10], 0))

    fname = os.path.join(dname, f"planes/sample_collection_simch")
    np.save(fname, np.stack(csimch[:10], 0))


def crop_planes_and_dump(dir_name, n_crops, patch_size, p):
    for s in ["induction", "collection"]:
        fname = os.path.join(dir_name, f"planes/{s}_clear.npy")
        cplanes = np.load(fname)[:, 0]

        fname = os.path.join(dir_name, f"planes/{s}_noisy.npy")
        nplanes = np.load(fname)[:, 0]

        medians = np.median(nplanes.reshape([nplanes.shape[0], -1]), axis=1)
        medians = medians.reshape([-1, 1, 1])
        m = (nplanes - medians).min()
        M = (nplanes - medians).max()

        # ensure nplanes are not just constant
        assert (M - m) != 0

        # normalize noisy planes
        nplanes = nplanes - medians

        ccrops = []
        ncrops = []
        for cplane, nplane in zip(cplanes, nplanes):
            idx = get_crop(cplane, n_crops=n_crops, patch_size=patch_size, p=p)
            ccrops.append(cplane[idx])
            ncrops.append(nplane[idx])

        ccrops = np.concatenate(ccrops, 0)[:, None]
        ncrops = np.concatenate(ncrops, 0)[:, None]

        print(f"\n{s} clear crops:", ccrops.shape)
        print(f"{s} noisy crops:", ncrops.shape)

        fname = os.path.join(dir_name, f"crops/{s}_clear_{patch_size[0]}_{p}")
        np.save(fname, ccrops)

        fname = os.path.join(dir_name, f"crops/{s}_noisy_{patch_size[0]}_{p}")
        np.save(fname, ncrops)

        # median normalization
        fname = os.path.join(dir_name, f"../{s}_mednorm")
        np.save(fname, [m, M])


def preprocess_main(dir_name, n_crops, crop_edge, percentage):
    patch_size = (crop_edge, crop_edge)
    for i in ["train/crops", "train/planes", "val/planes", "test/planes"]:
        if not os.path.isdir(os.path.join(dir_name, i)):
            os.mkdir(os.path.join(dir_name, i))

    for s in ["train", "test", "val"]:
        print(f"\n{s}:")
        dname = os.path.join(dir_name, s)
        get_planes_and_dump(dname)

    # save the normalization (this contain info from all the apas)
    for s in ["induction", "collection"]:
        fname = os.path.join(dir_name, f"train/planes/{s}_noisy.npy")
        n = np.load(fname).flatten()

        # MinMax
        fname = os.path.join(dir_name, f"{s}_minmax")
        np.save(fname, [n.min(), n.max()])

        # zscore
        fname = os.path.join(dir_name, f"{s}_zscore")
        np.save(fname, [n.mean(), n.std()])

    dname = os.path.join(dir_name, "train")
    crop_planes_and_dump(dname, n_crops, patch_size, percentage)
