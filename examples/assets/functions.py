"""This module implements utility functions for the ``onnx`` example."""
from time import time as tm
import shutil
import subprocess as sp
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dunedn.networks.abstract_net import AbstractNet
from dunedn.utils.utils import add_info_columns


def check_in_output_folder(folders: dict):
    """Creates output folder directory tree.

    Utility function to create the main output folder and its sub-directory
    structure. The user should hold permissions to write in it.

    This function also copies necessary files into the output directory.

    Parameters
    ----------
    folders: dict
        Dictionary containing folders to be created.
    """
    # create directories
    folders["cards"].mkdir(parents=True, exist_ok=True)
    folders["onnx_save"].mkdir(parents=True, exist_ok=True)
    folders["base_plot"].mkdir(parents=True, exist_ok=True)
    folders["pytorch_plot"].mkdir(exist_ok=True)
    folders["onnx_plot"].mkdir(exist_ok=True)

    # copy runcard
    runcard_path = Path("../runcards/default.yaml")
    shutil.copyfile(runcard_path, folders["cards"] / "runcard.yaml")
    shutil.copyfile(runcard_path, folders["cards"] / "runcard_default.yaml")

    # extract input tarball
    tarzip = "dunetpc_inspired_v09_p2GeV_rawdigits.tar.gz"
    sp.run(["tar", "-xzf", tarzip, "-C", folders["out"]])


def inference(model: AbstractNet, evt: np.ndarray, fname: Path, dev: str = None):
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
    dev: str
        Device hosting computation.

    Returns
    -------
    inference_time: float
        The elapsed time for inference.
    """
    start = tm()
    evt_dn = model.predict(evt, dev)
    inference_time = tm() - start

    # add info columns
    evt_dn = add_info_columns(evt_dn)

    # save inference outputs
    np.save(fname, evt_dn)
    return inference_time


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
        r"""Inspired to ProtoDUNE SP simulation
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
    plt.savefig(fname, bbox_inches="tight")  # , dpi=300)
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
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
    plt.title(
        r"""Inspired to ProtoDUNE SP simulation
    Collection wire, Raw waveform
    """,
        y=1.0,
        pad=-10,
    )
    plt.plot(wire, lw=0.3)
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
    plt.close()

    fname = outdir / "visual_clear_wire.png"
    plt.title(
        r"""Inspired to ProtoDUNE SP simulation
    Collection wire, Clear waveform
    """,
        y=1.0,
        pad=-10,
    )
    plt.plot(wire_target, lw=0.3, color="red")
    plt.savefig(fname, bbox_inches="tight")
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
    plt.close()
