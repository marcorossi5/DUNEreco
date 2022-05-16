"""This module implements utility functions for the ``onnx`` example."""
from time import time as tm
import shutil
import subprocess as sp
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dunedn.inference.hitreco import DnModel
from dunedn.inference.inference import thresholding_dn
from dunedn.networks.gcnn.utils import gcnn_inference_pass
from dunedn.networks.onnx.utils import gcnn_onnx_inference_pass
from dunedn.networks.utils import BatchProfiler
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
    folders["id_plot"].mkdir(parents=True, exist_ok=True)
    folders["pytorch_plot"].mkdir(exist_ok=True)
    folders["onnx_plot"].mkdir(exist_ok=True)

    # copy runcard
    runcard_path = Path("../runcards/default.yaml")
    shutil.copyfile(runcard_path, folders["cards"] / "runcard.yaml")
    shutil.copyfile(runcard_path, folders["cards"] / "runcard_default.yaml")

    # extract input tarball
    tarzip = "dunetpc_inspired_v08_p2GeV_rawdigits.tar.gz"
    sp.run(["tar", "-xzf", tarzip, "-C", folders["out"]])


def inference(model: DnModel, evt: np.ndarray, fname: Path, dev: str = None):
    """Makes inference on event and computes time.

    Saves the output file to `fname`.

    Parameters
    ----------
    model: AbstractNet
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

    # thresholding
    evt_dn = thresholding_dn(evt_dn)

    # add info columns
    evt_dn = add_info_columns(evt_dn)

    # save inference outputs
    np.save(fname, evt_dn)
    return inference_time


def compare_performance_onnx(
    pytorch_model: DnModel,
    onnx_model: DnModel,
    dev: str,
    batch_size_list: list,
    nb_batches: int = 100,
) -> pd.DataFrame:
    """Compares PyTorch and Onnx inference time for different batch sizes.

    Parameters
    ----------
    pytorch_model: DnModel
        The pytorch model to make inference
    onnx_model: DnModel
        The Onnx model to make inference
    dev: str
        Device hosting PyTorch computation.
    batch_size_list: list
        The list of batch sizes.
    nb_batches: int
        The number of batches to be passed to the network.

    Returns
    -------
    performance_dict: dict
        The dictionary containing profiled inference times.
    """
    input_shape = pytorch_model.cnetwork.input_shape

    performance = []

    for batch_size in batch_size_list:
        batched_input_shape = (nb_batches, batch_size) + input_shape
        inputs = (torch.randn(batched_input_shape), torch.randn(batched_input_shape))

        torch_pr = BatchProfiler()
        gcnn_inference_pass(
            zip(*inputs), pytorch_model.cnetwork, dev, profiler=torch_pr
        )
        torch_mean, torch_err = torch_pr.get_stats()

        onnx_pr = BatchProfiler()
        gcnn_onnx_inference_pass(zip(*inputs), onnx_model.cnetwork, profiler=onnx_pr)
        onnx_mean, onnx_err = onnx_pr.get_stats()
        performance.extend([torch_mean, onnx_mean, torch_err, onnx_err])

    iterables = [batch_size_list, ["mean", "err"]]
    index = pd.MultiIndex.from_product(iterables, names=["batch", "value"])
    df = pd.DataFrame(performance, index=index, columns=["torch", "onnx"])
    return df


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
    fname = outdir / "visual_denoised_wire.png"
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


def plot_comparison_catplot(
    data: pd.DataFrame,
    out_folder: Path,
    with_graphics: bool = False,
):
    """Categorical plot for PyTorch vs Onnx performance.

    Parameters
    ----------
    data: pd.DataFrame
        The dataframe with the collected inference timings.
    out_folder: Path
        The folder to save the categorical plot to.
    with_graphics: bool
        Wether to show matplotlib plots or not.
    """
    lang = data.index.levels[0]
    torch_mean = data.loc[(slice(None), "mean"), "torch"].to_numpy()
    torch_err = data.loc[(slice(None), "err"), "torch"].to_numpy()
    onnx_mean = data.loc[(slice(None), "mean"), "onnx"].to_numpy()
    onnx_err = data.loc[(slice(None), "err"), "onnx"].to_numpy()

    ind = np.arange(len(lang))
    width = 0.4

    fname = out_folder / "pytorch_onnx_performance_comparison.png"
    ax = plt.subplot()
    ax.bar(
        ind - width,
        torch_mean,
        width,
        yerr=torch_err,
        align="center",
        alpha=0.5,
        color="b",
        label="PyTorch",
    )
    ax.bar(
        ind,
        onnx_mean,
        width,
        yerr=onnx_err,
        align="center",
        alpha=0.5,
        color="r",
        label="Onnx",
    )
    ax.set(xticks=ind - width / 2, xticklabels=lang)

    rel_perc = 100 * (onnx_mean - torch_mean) / torch_mean

    maxes = np.maximum(onnx_mean + onnx_err, torch_mean + torch_err)

    for i, (v, m) in enumerate(zip(rel_perc, maxes)):
        if v > 0:
            sgn = "+"
        elif v == 0:
            sgn = ""
        else:
            sgn = "-"
        ax.text(
            i - width / 2,
            m + 0.1,
            f"{sgn}{v:.2f} %",
            ha="center",
            color="black",
            fontsize=10,
        )

    plt.xlabel("Batch size")
    plt.ylabel("Time $[s]$")
    plt.title("Inference time for different batch sizes")
    plt.legend(frameon=False)
    plt.savefig(fname, bbox_inches="tight", dpi=600)
    print(f"Saved image at {fname}")
    if with_graphics:
        plt.show()
    plt.close()
