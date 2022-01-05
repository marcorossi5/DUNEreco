"""
    This module reproduces the results obtained in arXiv:2103.01596
"""
import argparse
from pathlib import Path
from time import time as tm
import numpy as np
from dunedn.denoising.inference import (
    inference_main,
    add_arguments_inference,
    compare_performance_dn,
)


def main():
    parser = argparse.ArgumentParser(
        description="Denoising benchmark for arXiv:2103.01596",
        add_help="loads event from file, denoise it and computes performance metrics",
    )
    add_arguments_inference(parser)
    parser.add_argument(
        "-t",
        type=Path,
        required=True,
        metavar="TARGET",
        dest="target",
        help="path to the output event file",
    )
    args = parser.parse_args()
    target = np.load(args.target)
    # remove target to make inference work

    args.func(args.input, args.output, args.modeltype, args.ckpt)
    dn_evt = inference_main()


if __name__ == "__main__":
    start = tm()
    main()
    print(f"Program done in {tm()-start} s")
