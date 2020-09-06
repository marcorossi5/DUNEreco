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


def image_plots():
    pass


def main():
    metrics_plots()

    image_plots()


if __name__ == '__main__':
    start = tm.time()
    main()
    print(f'Program done in {tm.time()-start}')
