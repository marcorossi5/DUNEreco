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


def plot_image_sample(
    plane: np.ndarray, wire: int, outdir: Path, with_graphics: bool = False
):
    """Plots an APA plane.

    Parameters
    ----------
    plane: np.ndarray
        The reconstructed plane.
    wire: int
        The wire number in the APA plane.
    outdir: Path
        The directory where to save the plot.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    fname = outdir / "visual_collection_plane.png"
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
    cmap = plt.get_cmap("PuBu")
    plt.imshow(plane, aspect="auto", cmap=cmap)  # , vmin=vmin, vmax=vmax)
    plt.axhline(y=wire, color="orange", lw=1, alpha=0.6, linestyle="dashed")
    plt.colorbar()
    if with_graphics:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight", dpi=300)
        print(f"Saved image at {fname}")
    plt.close()


def plot_wire_sample(
    wire: np.ndarray,
    wire_target: np.ndarray,
    outdir: Path,
    with_graphics: bool = False,
):
    """Plots a reconstructed wavefunction and its target one.

    Parameters
    ----------
    wire: np.ndarray
        The reconstructed wavefunction.
    wire_target: np.ndarray
        The target wavefunction.
    outdir: Path
        The directory where to save the plots.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    plt.rcParams.update({"text.usetex": True})
    fname = outdir / "visual_noisy_wire.png"
    plt.title("Noisy waveform")
    plt.plot(wire, lw=0.3)
    if with_graphics:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight")
        print(f"Saved image at {fname}")

    fname = outdir / "visual_clear_wire.png"
    plt.title("Clear waveform")
    plt.plot(wire_target, lw=0.3, color="red")
    if with_graphics:
        plt.show()
    else:
        plt.savefig(fname, bbox_inches="tight")
        print(f"Saved image at {fname}")
    plt.close()


def plot_example(outdir: Path, with_graphics: bool = False):
    """Plots the first collection plane and a wire waveform from an event sample.

    Three plots are produced:

    - collection plane ADC heatmap
    - reconstructed waveform of a wire sample
    - target waveform

    Parameters
    ----------
    outdir: Path
        The output directory.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    file_clear = outdir / "p2GeV_cosmics_inspired_rawdigit_evt8.npy"
    evt = np.load(file_clear)[:, 2:]
    # file_clear.unlink()
    file_noisy = outdir / "p2GeV_cosmics_inspired_rawdigit_noiseoff_evt8.npy"
    evt_target = np.load(file_noisy)[:, 2:]
    # file_noisy.unlink()

    _, cplanes = evt2planes(evt)
    _, cplanes_target = evt2planes(evt_target)

    plane = cplanes[0, 0, 480:]
    plane_target = cplanes_target[0, 0, 480:]
    wire = 330

    # remove median value for noisy plane
    # median = np.median(plane)
    # plane_sub = plane - median

    # vmin = min(plane.min(), plane_target.min())
    # vmax = max(plane.max(), plane_target.max())

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
    plot_example(args.outdir)
    print(f"Program done in {tm()-start}s")
