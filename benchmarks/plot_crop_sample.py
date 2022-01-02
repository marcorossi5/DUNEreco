import os
import numpy as np
import time as tm
import matplotlib as mpl
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

# import analysis_roi module from analysis folder
import importlib
from pathlib import Path
root_folder = Path(os.environ.get("DUNEDN_PATH"))
spec = importlib.util.spec_from_file_location("analysis_roi", root_folder / "analysis/analysis_roi.py")
analysis_roi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis_roi)


def cmap():
    """
    Set a cmap to visualize raw data better
    """
    PuBu = mpl.cm.get_cmap("PuBu", 256)
    points = np.linspace(0, 1, 256)
    mymap = PuBu(points)

    def func(x):
        k = 5
        y = (1 - x) / (k * x + 1)
        return y[::-1]

    cdict = {
        "red": np.stack([func(points), mymap[:, 0], mymap[:, 0]], axis=1),
        "green": np.stack([func(points), mymap[:, 1], mymap[:, 1]], axis=1),
        "blue": np.stack([func(points), mymap[:, 2], mymap[:, 2]], axis=1),
    }

    return mpl.colors.LinearSegmentedColormap("dunemap", cdict)


def main():
    mpl.rcParams.update(analysis_roi.mpl_settings)
    mpl.rcParams.update(
        {
            "figure.figsize": [4.8, 4.8],
            "figure.titlesize": 18,
            "axes.titlesize": 16,
            "axes.labelsize": 16,
        }
    )

    clear = np.load("../datasets/denoising/test/planes/collection_clear.npy")[0, 0]
    noisy = np.load("../datasets/denoising/test/planes/collection_noisy.npy")[0, 0]

    roi_clear = np.copy(clear)
    mask = np.logical_and(roi_clear >= 0, roi_clear <= 3.5)
    roi_clear[mask] = 0
    roi_clear[~mask] = 1

    dunemap = cmap()

    cmap_mis = mpl.colors.ListedColormap(["white", "blue"])
    boundaries = [-0.5, 0.5, 1.5]
    norm_mis = mpl.colors.BoundaryNorm(boundaries, cmap_mis.N, clip=True)
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="white",
            label="No hit",
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

    fig = plt.figure()
    fig.tight_layout()
    fig.suptitle("ProtoDUNE-SP Simulation")
    ax = fig.add_subplot()
    ax.set_title(r"ROI target crop")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Wire Number")
    z = ax.imshow(roi_clear, cmap_mis)
    ax.set_xlim([5705, 5737])
    ax.set_ylim([922, 954])
    ax.legend(handles=legend_elements, loc="lower left")
    plt.savefig(
        "denoising/benchmarks/plots/roi_crop_sample_target.pdf",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()

    fig = plt.figure()
    fig.suptitle("ProtoDUNE-SP Simulation", x=0.44, y=0.92)
    fig.tight_layout()
    ax = fig.add_subplot()
    ax.set_title(r"DN target crop, ADC heatmap")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Wire Number")
    z = ax.imshow(clear, cmap=dunemap)
    ax.set_xlim([5705, 5737])
    ax.set_ylim([922, 954])
    fig.colorbar(z, ax=ax, shrink=0.5)
    # fig.colorbar(z, orientation="horizontal", pad=0.1, shrink=0.6)
    plt.savefig(
        "denoising/benchmarks/plots/dn_crop_sample_target.pdf",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()

    fig = plt.figure()
    fig.suptitle("ProtoDUNE-SP Simulation", x=0.44, y=0.92)
    fig.tight_layout()
    ax = fig.add_subplot()
    ax.set_title(r"ROI input crop, ADC heatmap")
    ax.set_xlabel(r"Time")
    ax.set_ylabel(r"Wire Number")
    z = ax.imshow(noisy, cmap=dunemap)
    ax.set_xlim([5705, 5737])
    ax.set_ylim([922, 954])
    # fig.colorbar(z, orientation="horizontal", pad=0.2, shrink=0.6)
    fig.colorbar(z, ax=ax, shrink=0.5)
    plt.savefig(
        "denoising/benchmarks/plots/roi_crop_sample_input.pdf",
        bbox_inches="tight",
        dpi=200,
    )
    ax.set_title(r"DN input crop, ADC heatmap")
    plt.savefig(
        "denoising/benchmarks/plots/dn_crop_sample_input.pdf",
        bbox_inches="tight",
        dpi=200,
    )
    plt.close()


if __name__ == "__main__":
    start = tm.time()
    main()
    print(f"Program done in {tm.time()-start}")
