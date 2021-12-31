""" This module compare results on test set of DN against Pandora recob::wires"""
from analysis_roi import set_ticks
import argparse
import numpy as np
from time import time
import matplotlib.pyplot as plt
import matplotlib as mpl
from operator import itemgetter

PARSER = argparse.ArgumentParser()
PARSER.add_argument(
    "--dirname",
    "-p",
    default="final",
    type=str,
    help="Directory containing results to plot, format: denoising/output/CNN_dn_<XXX>/final_test",
)


def cmap():
    """
    Set a cmap to visualize raw data better
    """
    PuBu = mpl.cm.get_cmap("PuBu", 256)
    points = np.linspace(0, 1, 256)
    mymap = PuBu(points)

    def func(x):
        k = 10
        y = (1 - x) / (k * x + 1)
        return y[::-1]

    cdict = {
        "red": np.stack([func(points), mymap[:, 0], mymap[:, 0]], axis=1),
        "green": np.stack([func(points), mymap[:, 1], mymap[:, 1]], axis=1),
        "blue": np.stack([func(points), mymap[:, 2], mymap[:, 2]], axis=1),
    }

    return mpl.colors.LinearSegmentedColormap("dunemap", cdict)


def metrics_list(dirname):
    dir_name = f"./denoising/output/CNN_dn_{dirname}/final_test/"
    dir_name_gc = f"./denoising/output/GCNN_dn_{dirname}/final_test/"
    dir_name_w = "./denoising/benchmarks/results/"

    fname = dir_name + "dn_test_metrics.npy"
    dn_test_metrics = np.load(fname)

    fname = dir_name_gc + "dn_test_metrics.npy"
    dn_test_metrics_gc = np.load(fname)

    fname = dir_name_w + "pandora_wires_metrics.npy"
    pandora_wires = np.load(fname)

    D = [(r"cnn", *dn_test_metrics[2:]), (r"gcnn", *dn_test_metrics_gc[2:])]
    # (r'Pandora', *pandora_wires.flatten())]

    return D


def bar_plot(lang, use, err, fname, label):
    ind = np.arange(len(lang))
    width = 0.8
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.barh(ind, use, width, xerr=err, align="center", alpha=0.5, color="r")
    ax.set(yticks=ind, yticklabels=lang)

    # ax.text(0.5, 0.5, 'PRELIMINARY',
    #        fontsize=60, color='gray', rotation=45,
    #        ha='center', va='center', alpha=0.5,
    #        transform=ax.transAxes)

    plt.xlabel(label, fontsize=22)
    plt.title(r"Final Evaluation", fontsize=20)
    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


def metrics_plots(dirname):
    D = metrics_list(dirname)
    Dsort = sorted(D, key=itemgetter(3), reverse=True)

    lang = [x[0] for x in Dsort]

    dir_name = "denoising/benchmarks/plots/"
    fname = dir_name + "dn_pandora_wires_ssim.pdf"
    use = [x[1] for x in Dsort]
    err = [x[2] for x in Dsort]
    bar_plot(lang, use, err, fname, r"SSIM")

    fname = dir_name + "dn_pandora_wires_psnr.pdf"
    use = [x[3] for x in Dsort]
    err = [x[4] for x in Dsort]
    bar_plot(lang, use, err, fname, r"pSNR on ADC counts")

    fname = dir_name + "dn_pandora_wires_mse.pdf"
    use = [x[5] for x in Dsort]
    err = [x[6] for x in Dsort]
    bar_plot(lang, use, err, fname, r"mse")


def image_arrays(dirname):
    dir_name = f"denoising/output/CNN_dn_{dirname}/final_test/"
    fname = dir_name + "dn_test_res.npy"
    dn = np.load(fname)[:, 0, 480:]

    dir_name = f"denoising/output/GCNN_dn_{dirname}/final_test/"
    fname = dir_name + "dn_test_res.npy"
    dn_gc = np.load(fname)[:, 0, 480:]

    dir_name = "../datasets/backup/test/planes/"
    fname = dir_name + "collection_clear.npy"
    clear = np.load(fname)[:, 0, 480:]

    dir_name = "../datasets/backup/test/planes/"
    fname = dir_name + "collection_noisy.npy"
    noisy = np.load(fname)[:, 0, 480:]

    dir_name = "denoising/benchmarks/results/"
    fname = dir_name + "pandora_collection_wires.npy"
    pandora_wires = np.load(fname)[:, 0, 480:]

    return [dn, dn_gc, pandora_wires], clear, noisy


def fit_constant(wires, clear):
    from ROOT import TGraph, TF1

    ratios = clear.sum(-1) / wires.sum(-1)
    masks = wires.sum(-1) != 0

    fits = []
    for ratio, mask in zip(ratios, masks):
        points = ratio[mask]
        shape = points.shape[0]
        g = TGraph(shape, np.arange(shape).astype(float), points)
        func = TF1("func", "[0]", 0, 10)
        fit = g.Fit("func", "SQC").Get()
        fits.append([fit.Parameter(0), fit.Error(0)])

    return np.array(fits)


def metric(target, dn, no_zero=False):
    """
    Computes the metric for DN and baseline comparison

    Parameters:
        target: numpy array, shape (n_planes, n_channels)
        dn: numpy array, same shape of target

    Returns:
        out: float
        metric: float, the reduced metric
        sigma: numpy array, shape (n_planes)
               a list of statistical uncertainties
    """
    out = np.abs(target - dn)
    if no_zero:
        mask = dn == 0
        out = out[mask]
    return out.mean(), out.var() / np.prod(target.shape)


def metric_nn(target, dn):
    mean, sigma = metric(target, dn)
    return mean, sigma ** 0.5


def metric_wcls(target, wcls, k, delta_k):
    """
    Computes the contribution on statistical error due to k

    Parameters:
        target: numpy array, shape (n_planes, n_channels)
        dn: numpy array, same shape of target

    Returns:
        metric: float
        sigma: numpy array, shape (n_planes)
               a list of statistical uncertainties
    """
    n_planes = target.shape[0]
    mean, var = metric(target, k * wcls, False)

    var_k = ((wcls.mean(-1) * delta_k / n_planes) ** 2).sum()
    return mean, (var + var_k) ** 0.5


def fit_and_plot_metrics(dn, clear):
    fits = fit_constant(dn[2], clear)

    k = fits[:, 0]

    target = clear.sum(-1) / 1e3
    cnn = dn[0].sum(-1) / 1e3
    gcnn = dn[1].sum(-1) / 1e3
    wcls = dn[2].sum(-1) / 1e3

    D = [
        ("cnn", *metric_nn(target, cnn)),
        ("gcnn", *metric_nn(target, gcnn)),
        ("Baseline", *metric_wcls(target, wcls, fits[:, 1:], fits[:, 1])),
    ]

    Dsort = sorted(D, key=itemgetter(1), reverse=True)

    lang = [x[0] for x in Dsort]

    dir_name = "denoising/benchmarks/plots/"
    fname = dir_name + "dn_pandora_wires_custom.pdf"
    use = [x[1] for x in Dsort]
    err = [x[2] for x in Dsort]
    bar_plot(
        lang, use, err, fname, r"{E}[$|$Target-Output$|$] on integrated ADC counts"
    )

    return target[0], cnn[0], gcnn[0], wcls[0] * k[0]


def plot_planes():
    pass


def plot_waveforms():
    pass


def image_plots(dirname):
    # uncomment to plot ratio raw::RawDigits/recob:Wire
    dn, clear, noisy = image_arrays(dirname)
    vmin = clear[0].min()
    vmax = clear[0].max()

    dunecmap = cmap()

    dir_name = "denoising/benchmarks/plots/"
    fname = dir_name + "dn_pandora_wires_adc.pdf"

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle(
        "ProtoDUNE-SP Simulation Preliminary\nDN final evaluation, denoised ADC heatmap",
        y=0.95,
    )
    gs = fig.add_gridspec(nrows=2, ncols=2, hspace=0.05, wspace=0.1)

    ax = plt.subplot(gs[0, 0])
    ax.set_ylabel("Target, Wire number", fontsize=16)
    z = ax.imshow(clear[0], vmin=vmin, vmax=vmax, cmap=dunecmap, aspect="auto")
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

    ax = plt.subplot(gs[0, 1])
    ax.set_ylabel("Baseline", fontsize=16)
    ax.imshow(dn[2][0], vmin=vmin, vmax=vmax, cmap=dunecmap, aspect="auto")
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

    ax = plt.subplot(gs[1, 0])
    ax.set_ylabel("CNN, Wire number", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    ax.imshow(dn[0][0], vmin=vmin, vmax=vmax, cmap=dunecmap, aspect="auto")
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
    z = ax.imshow(dn[1][0], vmin=vmin, vmax=vmax, cmap=dunecmap, aspect="auto")
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

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.01, 0.7])
    fig.colorbar(z, cax=cbar_ax)

    plt.savefig(fname, bbox_inches="tight", dpi=400)
    plt.close()

    target, cnn, gcnn, wcls = fit_and_plot_metrics(dn, clear)

    dir_name = "denoising/benchmarks/plots/"
    fname = dir_name + "dn_pandora_wires_integrated.pdf"

    fig = plt.figure(figsize=(12, 9))
    fig.suptitle("ProtoDUNE-SP Simulation Preliminary", fontsize=20, y=0.95)
    gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.05)

    ax = plt.subplot(gs[:3])
    ax.set_title("Collection Plane, Integrated Waveforms", fontsize=18)
    ax.set_ylabel("Integrated Waveform [A.U.]", fontsize=16)

    ax.plot(target, lw=1, alpha=0.8, color="grey", label="target")
    ax.plot(cnn, lw=0.3, color="orange", label="cnn")
    ax.plot(gcnn, lw=0.3, color="b", label="gcnn")
    ax.plot(wcls, lw=0.3, color="forestgreen", label="Normalized\nBaseline")

    ax.legend(frameon=False)
    ax.set_xlim([0, 480])
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

    ax = plt.subplot(gs[-1])
    ax.set_ylabel("Ratio", fontsize=16)
    ax.set_xlabel("Wire Number", fontsize=16)

    ax.plot(target / target, lw=1, alpha=0.8, color="grey")
    ax.plot(cnn / target, lw=0.3, color="orange")
    ax.plot(gcnn / target, lw=0.3, color="b")
    ax.plot(wcls / target, lw=0.3, color="forestgreen")
    ax.set_xlim([0, 480])
    ax.set_ylim([0.8, 1.2])
    ax = set_ticks(ax, "y", 0.8, 1.2, 3, div=2, d=1)
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

    plt.savefig(fname, bbox_inches="tight", dpi=300)
    plt.close()


def main(dirname):
    mpl.rcParams["text.usetex"] = True
    mpl.rcParams["savefig.format"] = "pdf"
    mpl.rcParams["figure.titlesize"] = 20
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["ytick.labelsize"] = 17
    mpl.rcParams["xtick.labelsize"] = 17
    mpl.rcParams["legend.fontsize"] = 14
    metrics_plots(dirname)

    image_plots(dirname)


if __name__ == "__main__":
    args = vars(PARSER.parse_args())
    start = time()
    main(**args)
    print(f"Program done in {time()-start}")
