# This file is part of DUNEdn by M. Rossi
from pathlib import Path
from glob import glob
import numpy as np
from dunedn.preprocessing.putils import get_crop
from dunedn.geometry.helpers import evt2planes
from dunedn.utils.utils import median_subtraction, load_yaml
from dunedn.preprocessing.putils import save_normalization_info


def add_arguments_preprocessing(parser):
    parser.add_argument("runcard", type=Path, help="yaml configcard path")
    parser.add_argument("--dir_name", type=Path, help="directory path to datasets")
    parser.add_argument(
        "--save_sample", action="store_true", help="extract a smaller dataset"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="print processing status information",
    )
    parser.set_defaults(func=preprocess)


def preprocess(args):
    p = load_yaml(args.runcard)
    preprocess_main(
        p["dataset_dir"],
        p["nb_crops"],
        p["crop_edge"],
        p["pct"],
        args.verbose,
        args.save_sample,
    )


def get_planes_and_dump(dname, verbose, save_sample):
    """
    Populates the "planes" subfolder of dname directory with numpy arrays of
    planes taken from events in the "events" subfolder. Planes arrays have
    shape=(N,C,H,W)

    Parameters
    ----------
        - dname: Path, path to train|val|test dataset subfolder
        - verbose: bool, wether to print status information
        - save_sample: bool, wether to save a smaller dataset from the original one
    """
    # TODO: this function could probably be shortened
    iclear = []
    inoisy = []
    isimch = []
    cclear = []
    cnoisy = []
    csimch = []

    paths_clear = glob((dname / "evts/*noiseoff*").as_posix())
    assert len(paths_clear) != 0

    print(f"[+] Fetching files from {dname}")
    for path_clear in paths_clear:
        path_noisy = Path(path_clear.replace("rawdigit_noiseoff", "rawdigit"))
        path_simch = Path(path_clear.replace("rawdigit_noiseoff", "simch_labels"))
        path_clear = Path(path_clear)

        if verbose:
            print(f"\t{path_clear.name}")
            print(f"\t{path_noisy.name}")
            print(f"\t{path_simch.name}")

        c = np.load(path_clear)[:, 2:]
        n = np.load(path_noisy)[:, 2:]
        s = np.load(path_simch)[:, 2:]

        induction_c, collection_c = evt2planes(c)
        iclear.append(induction_c)
        cclear.append(collection_c)

        induction_n, collection_n = evt2planes(n)
        inoisy.append(induction_n)
        cnoisy.append(collection_n)

        induction_s, collection_s = evt2planes(s)
        isimch.append(induction_s)
        csimch.append(collection_s)

    reshape = lambda x: x.reshape((-1,) + x.shape[2:])
    iclear = reshape(np.stack(iclear))
    cclear = reshape(np.stack(cclear))

    inoisy = reshape(np.stack(inoisy))
    cnoisy = reshape(np.stack(cnoisy))

    isimch = reshape(np.stack(isimch))
    csimch = reshape(np.stack(csimch))

    # at this point planes have shape=(nb_events,N,1,H,W)
    # with N being the number of induction|collection planes in each event

    print(f"[+] Saving planes to {dname}/planes")
    if verbose:
        print("\tCollection clear planes: ", cclear.shape)
        print("\tCollection noisy planes: ", cnoisy.shape)
        print("\tCollection sim::SimChannel planes: ", csimch.shape)
        print("\tInduction clear planes: ", iclear.shape)
        print("\tInduction noisy planes: ", inoisy.shape)
        print("\tInduction sim::SimChannel planes: ", isimch.shape)

    # stack all the planes from different events together
    save = lambda x, y: np.save(dname / f"planes/{x}", y)

    save("induction_clear", iclear)
    save("collection_clear", cclear)

    save("induction_noisy", inoisy)
    save("collection_noisy", cnoisy)

    save("induction_simch", isimch)
    save("collection_simch", csimch)

    if save_sample:
        # extract a small collection sample from dataset
        print(f"[+] Saving sample dataset to {dname}/planes")
        save("sample_collection_clear", cclear[:10])
        save("sample_collection_noisy", cnoisy[:10])
        save("sample_collection_simch", csimch[:10])


def crop_planes_and_dump(dir_name, nb_crops, crop_size, pct, verbose):
    """
    Populates the "crop" folder: for each plane stored in `dir_name/planes` generate
    nb_crops of size crop_size. The value of pct fixes the signal / background
    crops balancing.

    Parameters
    ----------
        - dir_name: Path, directory path to datasets
        - nb_crops: int, number of crops from a single plane
        - crop_size: list, crop [height, width]
        - pct: float, signal / background crops balancing
        - verbose: bool, wether to print status information
    """
    for s in ["induction", "collection"]:

        fname = dir_name / f"planes/{s}_clear.npy"
        cplanes = np.load(fname)[:, 0]

        fname = dir_name / f"planes/{s}_noisy.npy"
        nplanes = np.load(fname)

        print(f"[+] Cropping {s} planes at {fname}")

        nplanes = median_subtraction(nplanes)[:, 0]

        ccrops = []
        ncrops = []
        for cplane, nplane in zip(cplanes, nplanes):
            idx = get_crop(cplane, nb_crops=nb_crops, crop_size=crop_size, pct=pct)
            ccrops.append(cplane[idx][:, None])
            ncrops.append(nplane[idx][:, None])

        ccrops = np.concatenate(ccrops, 0)
        ncrops = np.concatenate(ncrops, 0)

        fname = dir_name / f"crops/{s}_noisy_{crop_size[0]}_{pct}"
        print(f"Saving crops to {dir_name}")
        if verbose:
            print(f"{s} clear crops:", ccrops.shape)
            print(f"{s} noisy crops:", ncrops.shape)
        np.save(fname, ncrops)

        fname = dir_name / f"crops/{s}_clear_{crop_size[0]}_{pct}"
        np.save(fname, ccrops)


def preprocess_main(dir_name, nb_crops, crop_edge, pct, verbose, save_sample):
    crop_size = (crop_edge, crop_edge)
    for folder in ["train", "val", "test"]:
        dname = dir_name / folder
        (dname / "planes").mkdir(parents=True, exist_ok=True)
        if folder == "train":
            (dname / "crops").mkdir(exist_ok=True)
        get_planes_and_dump(dname, verbose, save_sample)
    for channel in ["induction", "collection"]:
        save_normalization_info(dir_name, channel)
    crop_planes_and_dump(dir_name / "train", nb_crops, crop_size, pct, verbose)
