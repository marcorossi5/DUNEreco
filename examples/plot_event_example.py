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
import matplotlib.pyplot as plt
from dunedn.geometry.helpers import evt2planes


def main(outdir: Path):
    """
    Plots the first collection plane and a wire waveform (noisy and ground
    truths) from an event sample.

    Parameters
    ----------
        - outdir: the output directory
    """
    file_clear = outdir / "p2GeV_cosmics_inspired_rawdigit_evt8.npy"
    evt = np.load(file_clear)[:, 2:]
    file_clear.unlink()
    file_noisy = outdir / "p2GeV_cosmics_inspired_rawdigit_noiseoff_evt8.npy"
    evt_target = np.load(file_noisy)[:, 2:]
    file_noisy.unlink()

    iplanes, cplanes = evt2planes(evt)
    iplanes_target, cplanes_target = evt2planes(evt_target)

    plane = cplanes[0, 0, 480:]
    plane_target = cplanes_target[0, 0, 480:]
    wire = 330

    cmap = plt.get_cmap("PuBu")

    # remove median value for noisy plane
    # median = np.median(plane)
    # plane_sub = plane - median

    # vmin = min(plane.min(), plane_target.min())
    # vmax = max(plane.max(), plane_target.max())

    plt.rcParams.update({"text.usetex": True})
    plt.title(
        r"""ProtoDUNE SP simulation, \texttt{ dunetpc v09\_10\_00}
    Collection plane, ADC heatmap
    """,
        y=1.0,
        pad=-10,
    )
    # TODO: use a divergent cmap to show what's above/below 0 ADC (spikes/negative
    # tails)
    # use a sequential cmap giving more importance to higher values to highlight
    # spikes
    plt.imshow(plane, aspect="auto", cmap=cmap)  # , vmin=vmin, vmax=vmax)
    plt.axhline(y=wire, color="orange", lw=1, alpha=0.6, linestyle="dashed")
    plt.colorbar()
    plt.savefig(outdir / "visual_collection_plane.png", bbox_inches="tight", dpi=300)
    plt.close()

    plt.title("Noisy waveform")
    plt.plot(plane[wire], lw=0.3)
    plt.savefig(outdir / "visual_noisy_wire.png", bbox_inches="tight")
    plt.close()

    plt.title("Clear waveform")
    plt.plot(plane_target[wire], lw=0.3, color="red")
    plt.savefig(outdir / "visual_clear_wire.png", bbox_inches="tight")
    plt.close()


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
    main(args.outdir)
    print(f"Program done in {tm()-start}s")
