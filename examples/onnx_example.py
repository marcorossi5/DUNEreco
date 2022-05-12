"""
    The script version of the ``onnx_example.ipynb`` jupyter notebook.

    Usage

    .. code-block:: bash

        python onnx_example.py    
"""
from pathlib import Path
import argparse
import numpy as np
from plot_event_example import plot_example
from assets.functions import check_in_output_folder, inference
from dunedn.inference.hitreco import DnModel
from dunedn.inference.analysis import analysis_main
from dunedn.utils.utils import load_runcard

# TODO: the check-in is not complete if runcard do not get copied into the tmp folder


def main(modeltype, version, pytorch_dev):
    # base folders
    base_folder = Path("../../output/tmp")
    ckpt_folder = Path(f"../saved_models/{modeltype}_{version}")

    # relative folders
    folders = {
        "base": base_folder,
        "out": base_folder / "models/onnx",
        "ckpt": ckpt_folder,
        "cards": base_folder / f"cards",
        "onnx_save": base_folder / f"models/onnx/saved_models/{modeltype}_{version}",
        "plot": base_folder / "models/onnx/plots",
        "base_plot": base_folder / "models/onnx/plots/inputs",
        "pytorch_plot": base_folder / "models/onnx/plots/torch",
        "onnx_plot": base_folder / "models/onnx/plots/onnx",
    }

    check_in_output_folder(folders)

    # path to files
    paths = {
        "input": folders["out"] / "p2GeV_cosmics_inspired_rawdigit_evt8.npy",
        "target": folders["out"] / "p2GeV_cosmics_inspired_rawdigit_noiseoff_evt8.npy",
        "pytorch": folders["out"]
        / f"p2GeV_cosmics_inspired_rawdigit_torch_{modeltype}_evt8.npy",
        "onnx": folders["out"]
        / f"p2GeV_cosmics_inspired_rawdigit_onnx_{modeltype}_evt8.npy",
    }

    plot_example(paths["input"], paths["target"], outdir=folders["base_plot"])

    evt = np.load(paths["input"])[:, 2:]
    print(f"Loaded event at {paths['input']}")

    ############################################################################
    # PyTorch
    setup = load_runcard(base_folder / "cards/runcard.yaml")  # settings
    model = DnModel(setup, modeltype, ckpt_folder)
    print(f"Loaded model from {ckpt_folder} folder")

    pytorch_time = inference(model, evt, pytorch_dev, paths["pytorch"])
    print(f"PyTorch inference done in {pytorch_time}s")

    analysis_main(paths["pytorch"], paths["target"])
    plot_example(paths["pytorch"], paths["target"], outdir=folders["pytorch_plot"])

    ############################################################################
    # Onnx
    model.onnx_export(folders["onnx_save"])
    model_onnx = DnModel(setup, modeltype, folders["onnx_save"], should_use_onnx=True)
    onnx_time = inference(model_onnx, evt, paths["onnx"])
    print(f"ONNX inference done in {onnx_time}s")

    analysis_main(paths["onnx"], paths["target"])
    plot_example(paths["onnx"], paths["target"], outdir=folders["onnx_plot"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", "-d", help="device hosting the computation")
    parser.add_argument("--model", "-m", help="modeltype", dest="modeltype")
    parser.add_argument("--version", "-v", help="model version", default="v08")
    args = parser.parse_args()
    main(args.modeltype, args.version, args.dev)
