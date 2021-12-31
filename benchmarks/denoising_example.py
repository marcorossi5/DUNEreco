# TODO: transform this example into a jupyter notebook for further inspection
# TODO: think about exposing functions

import argparse
from time import time as tm
from dunedn.denoising.inference import inference_main, add_arguments_inference
from dunedn.denoising.hitreco import compute_metrics


def main():
    dn_evt = inference_main()


if __name__ == "__main__":

    start = tm()
    main()
    print(f"Program done in {tm()-start} s")
