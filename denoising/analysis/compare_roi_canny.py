""" This module compare results on test set of ROI against Canny filters"""
import sys
import os
import argparse
import numpy as np
import time as tm
import matplotlib.pyplot as plt
import matplotlib as mpl
from operator import itemgetter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dataloader import PlaneLoader
from utils.utils import compute_psnr
from losses import loss_ssim, loss_mse


def metrics_list():
    dir_name = './denoising/output/CNN_dn_final/final_test/'
    dir_name_gc = './denoising/output/GCNN_dn_final/final_test/'
    dir_name_c = './denoising/benchmarks/results/'

    fname = dir_name + 'roi_test_metrics.npy'
    #roi_test_metrics = np.load(fname)
    roi_test_metrics = [0 for i in range(8)]

    fname = dir_name_gc + 'roi_test_metrics.npy'
    #roi_test_metrics_gc = np.load(fname)
    roi_test_metrics_gc = [0 for i in range(8)]

    fname = dir_name_c + 'canny_metrics.npy'
    c_metrics = np.load(fname)

    D = [(r'cnn'  , *roi_test_metrics),
         (r'gcnn' , *roi_test_metrics_gc),
         (r'Canny', *c_metrics.flatten())]

    return D

def set_ticks(ax, start, end, numticks, axis, nskip=2):
    """
    Set both major and minor axes ticks in the logarithmical scale
    Parameters:
        ax: matplotlib.axes.Axes object
        start: int, leftmost tick
        end: int, rightmost tick
        numticks
        axis: 1 y axis, 0 x axis
        nskip: int, major ticks to leave without label
    """

    ticks = list(np.logspace(start,end,end-start+1))
    labels = [r'$10^{%d}$'%start]
    for i in [i for i in range(start+2,end+1,nskip)]:
        labels.extend(['' for i in range(nskip-1)]+[r'$10^{%d}$'%i])
    locmin = mpl.ticker.LogLocator(base=10.0,
                                   subs=[i/10 for i in range(1,10)],
                                   numticks=numticks)
    if axis == 'x':
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(locmin)
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    return ax


def bar_plot(lang, use, err, fname, label, log=False):
    """
    Parameters:
        log: bool, if plot x axis in log scale
    """
    ind = np.arange(len(lang))
    width=0.8

    ax = plt.subplot(111)
    ax.barh(ind, use, width, xerr=err , align='center', alpha=0.5, color='r')
    ax.set(yticks=ind, yticklabels=lang)
    if log:
        ax.set_xscale('log')

    ax.tick_params(axis='x', which='both', direction='in')

    plt.xlabel(label)
    plt.title(r'Final Evaluation')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()


def metrics_plots():
    D = metrics_list()
    Dsort = sorted(D, key=itemgetter(3), reverse=True)

    lang = [x[0] for x in Dsort]


    dir_name = 'denoising/benchmarks/plots/'
    fname = dir_name + 'roi_canny_acc.pdf'
    use  = [1-x[1] for x in Dsort]
    err = [x[2] for x in Dsort]
    bar_plot(lang, use, err, fname,
             r'$1-$ Accuracy', True)
    print(use)

    fname = dir_name + 'roi_canny_sns.pdf'
    use  = [x[3] for x in Dsort]
    err = [x[4] for x in Dsort]
    bar_plot(lang, use, err, fname, r'Sensitivity')
    print(use)

    fname = dir_name + 'roi_canny_spc.pdf'
    use  = [1-x[5] for x in Dsort]
    err = [x[6] for x in Dsort]
    bar_plot(lang, use, err, fname,
             r'$1-$ Specificity', True)
    print(use)

    fname = dir_name + 'roi_canny_auc.pdf'
    use  = [1-x[7] for x in Dsort]
    err = [x[8] for x in Dsort]
    bar_plot(lang, use, err, fname,
             r'$1-$ Area Under Curve', True)
    print(use)


def image_plots():
    pass


def main():
    metrics_plots()

    image_plots()


if __name__ == '__main__':
    start = tm.time()
    main()
    print(f'Program done in {tm.time()-start}')
