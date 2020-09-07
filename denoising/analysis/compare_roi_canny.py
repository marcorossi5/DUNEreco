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

from analysis_roi import set_ticks


def metrics_list():
    dir_name = './denoising/output/CNN_dn_final/final_test/'
    dir_name_gc = './denoising/output/GCNN_dn_final/final_test/'
    dir_name_c = './denoising/benchmarks/results/'

    fname = dir_name + 'roi_test_metrics.npy'
    roi_test_metrics = np.load(fname)

    fname = dir_name_gc + 'roi_test_metrics.npy'
    roi_test_metrics_gc = np.load(fname)

    fname = dir_name_c + 'canny_metrics.npy'
    c_metrics = np.load(fname)

    D = [(r'cnn'  , *roi_test_metrics),
         (r'gcnn' , *roi_test_metrics_gc),
         (r'Canny', *c_metrics.flatten())]
    # metrics: acc, sns, spc, auc
    return D
'''
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
'''

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
    ax.set_xlim([0,1])
    ax = set_ticks(ax,'x',0,1,6,div=4, d=1)

    plt.xlabel(label)
    plt.title(r'Final Evaluation')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()


def metrics_plots():
    D = metrics_list()
    Dsort = sorted(D, key=itemgetter(3), reverse=True)
    #print(Dsort)

    lang = [x[0] for x in Dsort]

    dir_name = 'denoising/benchmarks/plots/'
    '''
    fname = dir_name + 'roi_canny_acc.pdf'
    use  = [x[1] for x in Dsort]
    err = [x[2] for x in Dsort]
    bar_plot(lang, use, err, fname,
             r'$1-$ Accuracy', True)
    '''

    fname = dir_name + 'roi_canny_sns.pdf'
    use  = [x[3] for x in Dsort]
    err = [x[4] for x in Dsort]
    bar_plot(lang, use, err, fname, r'Sensitivity')

    '''
    fname = dir_name + 'roi_canny_spc.pdf'
    use  = [x[5] for x in Dsort]
    err = [x[6] for x in Dsort]
    bar_plot(lang, use, err, fname,
             r'$1-$ Specificity', True)

    fname = dir_name + 'roi_canny_auc.pdf'
    use  = [x[7] for x in Dsort]
    err = [x[8] for x in Dsort]
    bar_plot(lang, use, err, fname,
             r'$1-$ Area Under Curve', True)
    '''


def image_arrays():
    dir_name = 'denoising/output/CNN_dn_final/final_test/'
    fname = dir_name + 'roi_test_res.npy'
    roi = np.load(fname)[0,0]

    dir_name = 'denoising/output/GCNN_dn_final/final_test/'
    fname = dir_name + 'roi_test_res.npy'
    roi_gc = np.load(fname)[0,0]

    dir_name = '../datasets/denoising/test/planes/'
    fname = dir_name + 'collection_clear.npy'
    clear = np.load(fname)[0,0]
    clear[clear!=0] = 1

    dir_name = '../datasets/denoising/test/planes/'
    fname = dir_name + 'collection_noisy.npy'
    noisy = np.load(fname)[0,0]

    dir_name = 'denoising/benchmarks/results/'
    fname = dir_name + 'canny_res.npy'
    canny = np.load(fname)

    return [roi, roi_gc, canny], clear, noisy


def image_plots():
    roi, clear, noisy = image_arrays()

    dir_name = 'denoising/benchmarks/plots/'
    fname = dir_name + 'roi_res_plot.pdf'

    fig = plt.figure()
    fig.suptitle('ROI final evaluation')
    gs = fig.add_gridspec(nrows=4, ncols=1, hspace=0.05)

    ax = plt.subplot(gs[0])
    ax.set_ylabel('Target')
    ax.imshow(clear, vmin=0, vmax=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = plt.subplot(gs[1])
    ax.set_ylabel('CNN')
    ax.imshow(roi[0], vmin=0, vmax=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = plt.subplot(gs[2])
    ax.set_ylabel('GCNN')
    z = ax.imshow(roi[1], vmin=0, vmax=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = plt.subplot(gs[3])
    ax.set_ylabel('Canny')
    ax.imshow(roi[2], vmin=0, vmax=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    fig.subplots_adjust(right=0.825)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.015, 0.7])
    fig.colorbar(z, cax=cbar_ax)
    
    plt.savefig(fname, bbox_inches='tight', dpi=400)
    plt.close()


def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.titlesize'] = 20
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    #metrics_plots()

    image_plots()


if __name__ == '__main__':
    start = tm.time()
    main()
    print(f'Program done in {tm.time()-start}')
