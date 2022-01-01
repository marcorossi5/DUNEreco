""" Plot roi confusion metrics statistics as a function of the threshold"""
from time import time as tm
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from analysis_roi import confusion_matrix, testing_res, mpl_settings


def main():
    mpl.rcParams.update(mpl_settings)
    mpl.rcParams["axes.titlesize"] = 18
    mpl.rcParams["axes.labelsize"] = 16
    mpl.rcParams["legend.fontsize"] = 13

    sns_all = []
    spc_all = []

    thresholds = [i / 2 for i in range(14)]

    for threshold in thresholds:
        x = testing_res("final", threshold)

        mask = x[0].astype(bool)  # everything non zero is True

        hit = []
        no_hit = []
        hit_gc = []
        no_hit_gc = []

        sns = []
        spc = []

        for m, pred, pred_gc in zip(mask, x[1], x[2]):
            hit = pred[m]
            no_hit = pred[~m]

            hit_gc = pred_gc[m]
            no_hit_gc = pred_gc[~m]

            tp, fp, fn, tn = confusion_matrix(hit, no_hit)
            tp_gc, fp_gc, _, tn_gc = confusion_matrix(hit_gc, no_hit_gc)
            sns.append([tp / (tp + fn), tp_gc / (tp + fn)])
            spc.append([1 - fp / (tn + fp), 1 - fp_gc / (tn_gc + fp_gc)])
        sns_all.append(sns)
        spc_all.append(spc)

    sns_all = np.array(sns_all)
    spc_all = np.array(spc_all)

    sns_mean = sns_all.mean(1)
    sns_std = sns_all.std(1) / sns_all.shape[1]
    spc_mean = spc_all.mean(1)
    spc_std = spc_all.std(1) / spc_all.shape[1]

    fig = plt.figure()
    fig.suptitle("ProtoDUNE-SP Simulation Preliminary")

    ax = fig.add_subplot()
    ax.set_title("Final evaluation: threshold impact")
    ax.step([], [], "", color="white", label="Sensitivity:")
    ax.step(thresholds, sns_mean[:, 0], color="lime", label="cnn", lw=0.5, where="post")
    ax.fill_between(
        thresholds,
        sns_mean[:, 0] - sns_std[:, 0],
        sns_mean[:, 0] + sns_std[:, 0],
        alpha=0.5,
        step="post",
        color="lime",
    )
    ax.step(
        thresholds, sns_mean[:, 1], color="coral", label="gcnn", lw=0.5, where="post"
    )
    ax.fill_between(
        thresholds,
        sns_mean[:, 1] - sns_std[:, 1],
        sns_mean[:, 1] + sns_std[:, 1],
        alpha=0.5,
        step="post",
        color="coral",
    )

    ax.step([], [], "", color="white", label="Specificity:")
    ax.step(
        thresholds,
        spc_mean[:, 0],
        color="forestgreen",
        linestyle="--",
        label="cnn",
        lw=0.5,
        where="post",
    )
    ax.fill_between(
        thresholds,
        spc_mean[:, 0] - spc_std[:, 0],
        spc_mean[:, 0] + spc_std[:, 0],
        alpha=0.5,
        step="post",
        color="lime",
    )
    ax.step(
        thresholds,
        spc_mean[:, 1],
        color="darkred",
        linestyle="--",
        label="gcnn",
        lw=0.5,
        where="post",
    )
    ax.fill_between(
        thresholds,
        spc_mean[:, 0] - spc_std[:, 0],
        spc_mean[:, 0] + spc_std[:, 0],
        alpha=0.5,
        step="post",
        color="coral",
    )
    ax.axvline(3.5, lw=0.3, linestyle="--", color="lightslategrey")
    ax.axhline(0.975, lw=0.3, linestyle="--", color="lightslategrey")
    ax.grid(lw=0.3, alpha=0.2, color="lightslategrey")
    ax.set_xlabel("Threshold [ADC]")
    ax.legend(frameon=False, ncol=2)
    plt.savefig("denoising/benchmarks/plots/roi_timpact.pdf", bbox="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    start = tm()
    main()
    print(f"Program done in {tm()-start}")
