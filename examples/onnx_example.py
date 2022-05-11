import argparse
from time import time as tm
from pathlib import Path
import numpy as np
from plot_event_example import plot_example
from dunedn.inference.hitreco import DnModel
from dunedn.inference.analysis import analysis_main
from dunedn.utils.utils import load_runcard


def inference(model, evt, fname):
    """Makes inference on event and computes time.

    Saves the output file to `fname`.

    Parameters
    ----------
    model: DnModel
        The pytorch or onnx based model.
    evt: np.ndarray
        The input raw data.
    fname: Path
        The output file name.

    Returns
    -------
    inference_time: float
        The elapsed time for inference.
    """
    start = tm()
    evt_dn = model.predict(evt)
    inference_time = tm() - start

    # add info columns
    nb_channels, _ = evt_dn.shape
    channels_col = np.arange(nb_channels).reshape([-1, 1])
    event_col = np.zeros_like(channels_col)
    evt_dn = np.concatenate([event_col, channels_col, evt_dn], axis=1)

    # save pytorch inference
    np.save(fname, evt_dn)
    return inference_time


def main(modeltype, dev):
    version = "v08"
    basedir = Path("../../output/tmp")
    outdir = basedir / "models/onnx"
    ckpt = Path(f"../saved_models/{modeltype}_{version}")  # folder with checkpoints
    setup = load_runcard(basedir / "cards/runcard.yaml") # settings runcard

    input_path = outdir / "p2GeV_cosmics_inspired_rawdigit_evt8.npy"
    evt = np.load(input_path)[:, 2:]
    print(f"Loaded event at {input_path}")

    model = DnModel(setup, modeltype, ckpt, dev)
    print(f"Loaded model from {ckpt} folder")

    target_path = outdir / "p2GeV_cosmics_inspired_rawdigit_noiseoff_evt8.npy"
    reco_path = outdir / "pytorch_inference_results.npy"
    pytorch_time = inference(model, evt, reco_path)
    print(f"PyTorch inference done in {pytorch_time}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", "-d", help="device hosting the computation")
    parser.add_argument("--model", "-m", help="modeltype", dest="modeltype")
    args = parser.parse_args()
    main(args.modeltype, args.dev)