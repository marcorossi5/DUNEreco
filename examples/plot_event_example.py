"""
    This module plots the first collection plane rawdigits, along with noisy and
    clear waveforms taken from `dunetpc_inspired_p2GeV_cosmics_rawdigits.npz`
    sample event.

    Usage:

    ```bash
    python plot_event_example.py --outdir <output folder>
    ```
"""
import argparse
from pathlib import Path
from time import time as tm
import numpy as np
from assets.functions import plot_image_sample, plot_wire_sample
from dunedn.geometry.helpers import evt2planes


def plot_example(
    input_path: Path,
    target_path: Path,
    outdir: Path = None,
    with_graphics: bool = False,
):
    """Plots the first collection plane and a wire waveform from an event sample.

    Three plots are produced:

    - collection plane ADC heatmap
    - reconstructed waveform of a wire sample
    - target waveform

    The plots are saved in the ``outdir`` directory.
    If ``with_graphics`` is ``True``, plots are shown in the matplotlib GUI.

    Parameters
    ----------
    input_path: Path
        The path to the noisy inputs file.
    target_path: Path
        The path to the clear target file.
    outdir: Path
        Directory to save plots into. Defaults to the same directory of
        `input_path`.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    if outdir is None:
        outdir = input_path.parent
    evt = np.load(input_path)[:, 2:]
    # file_clear.unlink()
    evt_target = np.load(target_path)[:, 2:]
    # file_noisy.unlink()

    _, cplanes = evt2planes(evt)
    _, cplanes_target = evt2planes(evt_target)

    plane = cplanes[0, 0, 480:]
    plane_target = cplanes_target[0, 0, 480:]
    wire = 330

    plot_image_sample(plane, wire, outdir, with_graphics)

    plot_wire_sample(plane[wire], plane_target[wire], outdir, with_graphics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("plot event")
    parser.add_argument(
        "--output",
        type=Path,
        help="the output directory",
        default="plot_example",
        dest="outdir",
    )
    args = parser.parse_args()
    start = tm()
    input_path = args.outdir / "p2GeV_cosmics_inspired_rawdigit_evt8.npy"
    target_path = args.outdir / "p2GeV_cosmics_inspired_rawdigit_noiseoff_evt8.npy"
    plot_example(input_path, target_path)
    print(f"Program done in {tm()-start}s")
