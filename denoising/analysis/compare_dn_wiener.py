""" This module compare results on test set of DN against Wiener filters"""
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
    dir_name_w = './denoising/benchmarks/results/'

    fname = dir_name + 'dn_test_metrics.npy'
    dn_test_metrics = np.load(fname)

    fname = dir_name_gc + 'dn_test_metrics.npy'
    dn_test_metrics_gc = np.load(fname)

    fname = dir_name_w + 'wiener_3_metrics.npy'
    w_3_metrics = np.load(fname)

    fname = dir_name_w + 'wiener_5_metrics.npy'
    w_5_metrics = np.load(fname)

    fname = dir_name_w + 'wiener_7_metrics.npy'
    w_7_metrics = np.load(fname)

    D = [(r'cnn'       , *dn_test_metrics[2:]),
         (r'gcnn'      , *dn_test_metrics_gc[2:]),
         (r'Wiener $3$', *w_3_metrics.flatten()),
         (r'Wiener $5$', *w_5_metrics.flatten()),
         (r'Wiener $7$', *w_7_metrics.flatten())]

    return D

def bar_plot(lang, use, err, fname, label):
    ind = np.arange(len(lang))
    width=0.8

    ax = plt.subplot(111)
    ax.barh(ind, use, width, xerr=err , align='center', alpha=0.5, color='r')
    ax.set(yticks=ind, yticklabels=lang)

    plt.xlabel(label)
    plt.title(r'Final Evaluation')
    plt.savefig(fname, bbox_inches='tight', dpi=300)
    plt.close()


def metrics_plots():
    D = metrics_list()
    Dsort = sorted(D, key=itemgetter(3), reverse=True)

    lang = [x[0] for x in Dsort]

    dir_name = 'denoising/benchmarks/plots/'
    fname = dir_name + 'dn_wiener_ssim.pdf'
    use  = [x[1] for x in Dsort]
    err = [x[2] for x in Dsort]
    bar_plot(lang, use, err, fname, r'SSIM')

    fname = dir_name + 'dn_wiener_psnr.pdf'
    use  = [x[3] for x in Dsort]
    err = [x[4] for x in Dsort]
    bar_plot(lang, use, err, fname, r'pSNR')

    fname = dir_name + 'dn_wiener_mse.pdf'
    use  = [x[5] for x in Dsort]
    err = [x[6] for x in Dsort]
    bar_plot(lang, use, err, fname, r'mse')


def image_arrays():
    dir_name = 'denoising/output/CNN_dn_final/final_test/'
    fname = dir_name + 'dn_test_res.npy'
    dn = np.load(fname)[0,0]

    dir_name = 'denoising/output/GCNN_dn_final/final_test/'
    fname = dir_name + 'dn_test_res.npy'
    dn_gc = np.load(fname)[0,0]

    dir_name = '../datasets/denoising/test/planes/'
    fname = dir_name + 'collection_clear.npy'
    clear = np.load(fname)[0,0]

    dir_name = '../datasets/denoising/test/planes/'
    fname = dir_name + 'collection_noisy.npy'
    noisy = np.load(fname)[0,0]

    dir_name = 'denoising/benchmarks/results/'
    fname = dir_name + 'wiener_3_res.npy'
    w_3 = np.load(fname)[0,0]

    fname = dir_name + 'wiener_5_res.npy'
    w_5 = np.load(fname)[0,0]

    fname = dir_name + 'wiener_7_res.npy'
    w_7 = np.load(fname)[0,0]

    return [dn, dn_gc, w_3, w_5, w_7], clear, noisy


def image_plots():
    dn, clear, noisy = image_arrays()

    dir_name = 'denoising/benchmarks/plots/'
    fname = dir_name + 'dn_wires.pdf'

    fig = plt.figure(figsize=(21,16.27))
    fig.suptitle('Denoising final evaluation')
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.05)

    ax = plt.subplot(gs[0])
    ax.set_ylabel('Wire with hits')
    ax.plot(clear[500], lw=1, alpha=0.8, color='grey', label='target')
    ax.plot(dn[0][500], lw=0.3, label='cnn', color='orange')
    ax.plot(dn[1][500], lw=0.3, label='gcnn', color='b')
    ax.plot(dn[2][500], lw=0.3, label='Wiener 3', color='forestgreen')
    ax.plot(dn[3][500], lw=0.3, label='Wiener 5', color='lime')
    ax.plot(dn[4][500], lw=0.3, label='Wiener 7', color='cyan')
    ax.legend(frameon=False)
    ax.set_xlim([0,6000])
    #ax = set_ticks(ax,'y', 0, .7, 5, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = plt.subplot(gs[1])
    ax.set_ylabel('Wire w/o hits')
    ax.set_xlabel('Time ticks')
    ax.plot(clear[0], lw=1, alpha=0.8, color='grey', label='target')
    ax.plot(dn[0][0], lw=0.3, label='cnn', color='orange')
    ax.plot(dn[1][0], lw=0.3, label='gcnn', color='b')
    ax.plot(dn[2][0], lw=0.3, label='Wiener 3', color='forestgreen')
    ax.plot(dn[3][0], lw=0.3, label='Wiener 5', color='lime')
    ax.plot(dn[4][0], lw=0.3, label='Wiener 7', color='cyan')
    ax.set_xlim([0,6000])
    ax.set_ylim([-1.5,1.5])
    ax = set_ticks(ax,'y', -1.5, 1.5, 5, div=4, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    plt.savefig(fname, bbox_inches='tight', dpi=300)
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
