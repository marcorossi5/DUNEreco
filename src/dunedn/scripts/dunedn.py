# This file is part of DUNEdn by M. Rossi
import argparse
from time import time as tm
from dunedn.preprocessing.preprocess import add_arguments_preprocessing
from dunedn.denoising.denoise_training import add_arguments_training
from dunedn.denoising.inference import add_arguments_inference


def main():
    parser = argparse.ArgumentParser(description="dunedn")

    subparsers = parser.add_subparsers()

    # preprocess dataset before training
    p_subparser = subparsers.add_parser(
        "preprocess",
        description="Preprocess dataset of protoDUNE events: dumps planes and training crops.",
    )
    add_arguments_preprocessing(p_subparser)

    # train
    t_subparser = subparsers.add_parser("train", description="Train model.")
    add_arguments_training(t_subparser)

    # inference
    dn_subparser = subparsers.add_parser(
        "inference", description="Load event and make inference with saved model."
    )
    add_arguments_inference(dn_subparser)

    args = parser.parse_args()

    start = tm()
    # execute parsed function
    args.func(args)
    print(f"Program done in {tm()-start} s")


# TODO: train subcommand missing
# TODO: deal with the distributed training in a separated sub-package folder
# TODO: (enachement) introduce the hpt (HyperParameterTuning) subcommand
