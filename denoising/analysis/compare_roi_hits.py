""" This module compare results on test set of ROI against Pandora recob::hits"""
from analysis_roi import set_ticks
from losses import loss_ssim, loss_mse
from utils.utils import compute_psnr
from dataloader import PlaneLoader
import sys
import os
import argparse
import numpy as np
import time as tm
import matplotlib.pyplot as plt
import matplotlib as mpl
from operator import itemgetter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--dirname",
    "-p",
    default="final",
    type=str,
    help="Directory containing results to plot, format: denoising/output/CNN_dn_<XXX>/final_test",
)
PARSER.add_argument(
    "--threshold",
    "-t",
    default=3.5,
    type=float,
    help="Threshold to distinguish signal/noise in labels",
)


def metrics_list(dirname):
    dir_name = f"./denoising/output/CNN_dn_{dirname}/final_test/"
    dir_name_gc = f"./denoising/output/GCNN_dn_{dirname}/final_test/"
    dir_name_c = "./denoising/benchmarks/results/"

    fname = dir_name + "roi_test_metrics.npy"
    roi_test_metrics = np.load(fname)

    fname = dir_name_gc + "roi_test_metrics.npy"
    roi_test_metrics_gc = np.load(fname)

    fname = dir_name_c + "pandora_hits_metrics.npy"
    c_metrics = np.load(fname)

    D = [
        (r"cnn", *roi_test_metrics),
        (r"gcnn", *roi_test_metrics_gc),
        (r"Baseline", *c_metrics.flatten()),
    ]
    # metrics: acc, sns, spc
    return D


def bar_plot(lang, use, err, fname, label, log=False):
    """
    Parameters:
        log: bool, if plot x axis in log scale
    """
    ind = np.arange(len(lang))
    width = 0.8

    ax = plt.subplot(111)
    ax.barh(ind, use, width, xerr=err, align="center", alpha=0.5, color="r")
    ax.set(yticks=ind, yticklabels=lang)
    if log:
        ax.set_xscale("log")
        ax.set_xlim([1e-4, 1e-2])
    else:
        ax.tick_params(axis="x", which="both", direction="in")
        ax.set_xlim([0, 1])
        ax = set_ticks(ax, "x", 0, 1, 6, div=4, d=1)

    plt.xlabel(label, fontsize=22)
    plt.title(r"Final Evaluation")
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


def metrics_plots(dirname):
    """
    Just plot sensitivity as figure of merit
    others quantities are biased due to dataset
    unbalance in hit/no-hit
    """
    D = metrics_list(dirname)
    Dsort = sorted(D, key=itemgetter(3), reverse=True)

    lang = [x[0] for x in Dsort]

    dir_name = "denoising/benchmarks/plots/"

    fname = dir_name + "roi_pandora_hits_sns.pdf"
    use = [x[3] for x in Dsort]
    err = [x[4] for x in Dsort]
    bar_plot(lang, use, err, fname, r"Sensitivity")

    fname = dir_name + "roi_pandora_hits_spc.pdf"
    use = [1 - x[5] for x in Dsort]
    err = [x[6] for x in Dsort]
    bar_plot(lang, use, err, fname, r"False Positive Rate", log=True)


def image_arrays(dirname, threshold):
    dir_name = f"denoising/output/CNN_roi_{dirname}/final_test/"
    fname = dir_name + "roi_test_res.npy"
    roi = np.load(fname)[0, 0]

    dir_name = f"denoising/output/GCNN_roi_{dirname}/final_test/"
    fname = dir_name + "roi_test_res.npy"
    roi_gc = np.load(fname)[0, 0]

    dir_name = "../datasets/backup/test/planes/"
    fname = dir_name + "collection_clear.npy"
    clear = np.load(fname)[0, 0]
    mask = np.logical_and(clear >= 0, clear <= threshold)
    clear[mask] = 0
    clear[~mask] = 1

    dir_name = "../datasets/backup/test/planes/"
    fname = dir_name + "collection_noisy.npy"
    noisy = np.load(fname)[0, 0]

    dir_name = "denoising/benchmarks/results/"
    fname = dir_name + "pandora_collection_hits.npy"
    canny = np.load(fname)[0, 0]

    return [roi, roi_gc, canny], clear, noisy


def image_plots(dirname, threshold):
    roi, clear, noisy = image_arrays(dirname, threshold)

    dir_name = "denoising/benchmarks/plots/"
    fname = dir_name + "roi_res_plot.pdf"

    from matplotlib.lines import Line2D

    cmap = mpl.colors.ListedColormap(["green", "white", "darkred"])
    boundaries = [-1, -0.5, 0.5, 1]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap.N, clip=True)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="False Negative",
            markerfacecolor="green",
            markeredgecolor="black",
            markersize=7,
        ),
        # Line2D([0], [0], marker='o', color='white', label='Correct',
        #   markerfacecolor='white', markeredgecolor='black', markersize=7),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="False Positive",
            markerfacecolor="darkred",
            markeredgecolor="black",
            markersize=7,
        ),
    ]

    cmap_target = mpl.colors.ListedColormap(["white", "blue"])
    boundaries = [-1, -0.5, 0.5, 1]
    norm = mpl.colors.BoundaryNorm(boundaries, cmap_target.N, clip=True)
    legend_target = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="No Hit",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=7,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="Hit",
            markerfacecolor="blue",
            markeredgecolor="black",
            markersize=7,
        ),
    ]

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        "ProtoDUNE-SP Simulation Preliminary:\nROI final evaluation, classification score"
    )
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.05, wspace=0.1)

    ax = plt.subplot(gs[0, 0])
    ax.set_ylabel("Target, Wire number", fontsize=16)
    ax.imshow(clear[480:], vmin=0, vmax=1, cmap=cmap_target, aspect="auto")
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        top=True,
        labeltop=False,
        bottom=True,
        labelbottom=False,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        right=True,
        labelright=False,
        left=True,
        labelleft=True,
    )
    ax.legend(handles=legend_target, ncol=2, loc=(0.455, 1.01))

    ax = plt.subplot(gs[0, 1])
    ax.set_ylabel("Baseline", fontsize=16)
    ax.imshow(roi[2][480:] - clear[480:], cmap=cmap, aspect="auto")
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        top=True,
        labeltop=False,
        bottom=True,
        labelbottom=False,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        right=True,
        labelright=False,
        left=True,
        labelleft=False,
    )
    ax.legend(handles=legend_elements, ncol=2, loc=(0.13, 1.01))

    ax = plt.subplot(gs[1, 0])
    ax.set_ylabel("CNN, Wire number", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    ax.imshow((roi[0][480:] > 0.5).astype(int) - clear[480:], cmap=cmap, aspect="auto")
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        top=True,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        right=True,
        labelright=False,
        left=True,
        labelleft=True,
    )

    ax = plt.subplot(gs[1, 1])
    ax.set_ylabel("GCNN", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    z = ax.imshow(
        (roi[1][480:] > 0.5).astype(int) - clear[480:], cmap=cmap, aspect="auto"
    )
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        top=True,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        right=True,
        labelright=False,
        left=True,
        labelleft=False,
    )

    plt.savefig(fname, bbox_inches="tight", dpi=400)
    plt.close()

    ##########################################################################
    dir_name = "denoising/benchmarks/plots/"
    fname = dir_name + f"roi_mismatch_plot_t{int(threshold)}.pdf"

    fig = plt.figure(figsize=(12, 5.4))
    suptitle = (
        f"ProtoDUNE-SP Simulation Preliminary\nMismatched points, $t= {int(threshold)}$"
    )
    fig.suptitle(r"%s" % suptitle, x=0.3)

    cmap_mis = mpl.colors.ListedColormap(["green", "white", "darkred"])
    boundaries = [-1.5, -0.5, 0.5, 1]
    norm_mis = mpl.colors.BoundaryNorm(boundaries, cmap_mis.N, clip=True)

    gs = fig.add_gridspec(nrows=1, ncols=2, hspace=0.05, wspace=0.15)

    ax = plt.subplot(gs[0, 0])
    ax.set_ylabel("CNN, Wire number", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    difference = (roi[0][480:] > 0.5).astype(int) - clear[480:]
    ax.imshow(difference, cmap=cmap_mis, norm=norm_mis, aspect="auto")
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        top=True,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        right=True,
        labelright=False,
        left=True,
        labelleft=True,
    )

    ax = plt.subplot(gs[0, 1])
    ax.set_ylabel("GCNN", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    difference = (roi[1][480:] > 0.5).astype(int) - clear[480:]
    z = ax.imshow(difference, cmap=cmap_mis, norm=norm_mis, aspect="auto")
    ax.tick_params(
        axis="x",
        which="both",
        direction="in",
        top=True,
        labeltop=False,
        bottom=True,
        labelbottom=True,
    )
    ax.tick_params(
        axis="y",
        which="both",
        direction="in",
        right=True,
        labelright=False,
        left=True,
        labelleft=True,
    )
    ax.legend(handles=legend_elements, ncol=2, loc=(0.112, 1.01))

    plt.savefig(fname, bbox_inches="tight", dpi=400)
    plt.close()


def main(dirname, threshold):
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["savefig.format"] = "pdf"
    mpl.rcParams["figure.titlesize"] = 20
    mpl.rcParams["axes.titlesize"] = 18
    mpl.rcParams["ytick.labelsize"] = 16
    mpl.rcParams["xtick.labelsize"] = 14
    mpl.rcParams["legend.fontsize"] = 14
    metrics_plots(dirname)

    image_plots(dirname, threshold)


if __name__ == "__main__":
    args = vars(PARSER.parse_args())
    start = tm.time()
    main(**args)
    print(f"Program done in {tm.time()-start}")
