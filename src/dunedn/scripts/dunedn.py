# This file is part of DUNEdn by M. Rossi
import argparse
from time import time as tm
from dunedn.preprocessing.preprocess import add_arguments_preprocessing
from dunedn.training.denoise_training import add_arguments_training
from dunedn.inference.inference import add_arguments_inference

def main():
    parser = argparse.ArgumentParser(description="dunedn")

    subparsers = parser.add_subparsers()

    # preprocess dataset before training
    p_msg = "Preprocess dataset of protoDUNE events: dumps planes and training crops."
    p_subparser = subparsers.add_parser(
        "preprocess", description=p_msg, help=p_msg.lower().split(":")[0]
    )
    add_arguments_preprocessing(p_subparser)

    # train
    t_msg = "Train model loading settings from configcard."
    t_subparser = subparsers.add_parser(
        "train", aliases=["training"], description=t_msg, help=t_msg.lower().strip(".")
    )
    add_arguments_training(t_subparser)

    # inference
    dn_msg = "Load event and make inference with saved model."
    dn_subparser = subparsers.add_parser(
        "inference",
        aliases=["infer"],
        description=dn_msg,
        help=dn_msg.lower().strip("."),
    )
    add_arguments_inference(dn_subparser)

    args = parser.parse_args()

    start = tm()
    # execute parsed function
    args.func(args)
    print(f"Program done in {tm()-start} s")


# TODO: train subcommand missing
# TODO: deal with the distributed training in a separated sub-package folder
# TODO: (enhancement) introduce the hpt (HyperParameterTuning) subcommand
