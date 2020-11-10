""" This module computes the Wiener filter for planes in the test set"""
import sys
import os
import argparse
import numpy as np
import time as tm
import matplotlib.pyplot as plt
import matplotlib as mpl
from analysis_roi import set_ticks, training_metrics,\
                         training_timings, special_ticks

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dirname", "-p", default="final",
                    type=str, help='Directory containing results to plot, format: denoising/output/CNN_dn_<XXX>/final_test')

def training_plots(dirname):
    loss, val_epochs, val_metrics = training_metrics('dn', dirname)
    epochs = [i for i in range(len(loss[0]))]

    fig = plt.figure()
    fig.suptitle('Training Loss')
    
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.2)
    ax = fig.add_subplot(gs[0])
    ax.set_title('Training')
    ax.set_ylabel(r'$(1-$ SSIM$)$ + MSE')
    ax.plot(epochs, loss[0], label='cnn', color='#ff7f0e')
    ax.plot(epochs, loss[1], label='gcnn', color='b')
    ax.set_xlim([0,50])
    ax = set_ticks(ax,'x', 0, 50, 6)
    ax.set_ylim([0,.4])
    ax = set_ticks(ax,'y', 0, .4, 5, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[1])
    ax.set_title('Validation')
    ax.errorbar(val_epochs[1], val_metrics[0][0], yerr=val_metrics[0][1],
                label='cnn', linestyle='--', color='#ff7f0e', marker='s')
    ax.errorbar(val_epochs[1], val_metrics[1][0], yerr=val_metrics[1][1],
                label='gcnn', linestyle='--', color='b', marker='^')
    ax.set_xlim([0,50])
    ax = set_ticks(ax,'x', 0, 50, 6)
    ax.set_ylim([0,8000])
    ax = set_ticks(ax,'y', 0, 8000, 5)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=True,
                   left=True, labelleft=False)

    plt.savefig(f'denoising/benchmarks/plots/training_loss_dn.pdf',
                bbox_inches='tight', dpi=250)
    plt.close()

    ##########################################################################

    fig = plt.figure()
    fig.suptitle('Validation Metrics')
    gs = fig.add_gridspec(nrows=3, ncols=1, hspace=0.2)

    ax = fig.add_subplot(gs[0])
    ax.set_ylabel('SSIM')
    ax.errorbar(val_epochs[0], val_metrics[0][2], yerr=val_metrics[0][3],
                label='val cnn', linestyle='--', color='#ff7f0e', marker='^')
    ax.errorbar(val_epochs[1], val_metrics[1][2], yerr=val_metrics[1][3],
                label='val gcnn', linestyle='--', color='b', marker='^')
    ax.legend(frameon=False)
    ax.set_xlim([0,50])
    ax = set_ticks(ax,'x', 0, 50, 6)
    ax.set_ylim([0.01,.07])
    ax = set_ticks(ax,'y', 0.01, .07, 3, d=2)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = fig.add_subplot(gs[1])
    ax.set_ylabel('pSNR')
    ax.errorbar(val_epochs[0], val_metrics[0][4], yerr=val_metrics[0][5],
                label='val cnn', linestyle='--', color='#ff7f0e', marker='^')
    ax.errorbar(val_epochs[1], val_metrics[1][4], yerr=val_metrics[1][5],
                label='val gcnn', linestyle='--', color='b', marker='^')
    ax.set_xlim([0,50])
    ax = set_ticks(ax,'x', 0, 50, 6)
    ax.set_ylim([40,70])
    ax = set_ticks(ax,'y', 40, 70, 4)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = fig.add_subplot(gs[2])
    ax.set_ylabel('MSE')
    ax.set_xlabel('Epoch')
    ax.errorbar(val_epochs[0], val_metrics[0][6], yerr=val_metrics[0][7],
                label='val cnn', linestyle='--', color='#ff7f0e', marker='^')
    ax.errorbar(val_epochs[1], val_metrics[1][6], yerr=val_metrics[1][7],
                label='val gcnn', linestyle='--', color='b', marker='^')
    ax.set_xlim([0,50])
    ax = set_ticks(ax,'x', 0, 50, 6)
    ax.set_ylim([0,50])
    ax = set_ticks(ax,'y', 0, 50, 3)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    plt.savefig(f'denoising/benchmarks/plots/training_metrics_dn.pdf',
                bbox_inches='tight', dpi=250)
    plt.close()

    ##########################################################################

    timings_train, timings_val = training_timings('dn', dirname)

    fig = plt.figure()
    fig.suptitle('Timings')
    gs = fig.add_gridspec(nrows=4, ncols=2, wspace=0.2, hspace=0.2)

    ax = fig.add_subplot(gs[:3,0])
    ax.set_title('Training')
    ax.set_ylabel('Time  per batch [s]')
    ax.plot(epochs, timings_train[0]/94, label='cnn', color='#ff7f0e')
    ax.plot(epochs, timings_train[1]/94, label='gcnn', color='b')
    ax.set_xlim([0,50])
    ax = set_ticks(ax,'x', 0, 50, 6)
    ax.set_ylim([0,4])
    ax = set_ticks(ax,'y', 0, 4, 5)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    ax.legend(frameon=False)

    ax = fig.add_subplot(gs[3,0])
    ax.set_ylabel('Ratio')
    ax.set_xlabel('Epoch')
    ax.plot(epochs, timings_train[1]/timings_train[0])
    ax.set_xlim([0,50])
    ax = set_ticks(ax,'x', 0, 50, 6)
    ax.set_ylim([3.5,5.5])
    ax = set_ticks(ax,'y', 3.5, 5.5, 3, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = fig.add_subplot(gs[:3,1])
    ax.set_title('Validation')
    ax.plot(val_epochs[0], timings_val[0]/136, label='cnn',
            color='#ff7f0e', linestyle='--')
    ax.plot(val_epochs[1], timings_val[1]/136, label='gcnn',
            color='b', linestyle='--')
    ax.set_xlim([5,50])
    ax = special_ticks(ax, 5, 50, [5,25,50])
    ax.set_ylim([0,2])
    ax = set_ticks(ax,'y', 0, 2, 5, div=4, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=True,
                   left=True, labelleft=False)

    ax = fig.add_subplot(gs[3,1])
    ax.set_xlabel('Epoch')
    ax.plot(val_epochs[0], timings_val[1]/timings_val[0], linestyle='--')
    ax.set_xlim([5,50])
    ax = special_ticks(ax, 5, 50, [5,10,20,30,40,50])
    ax.set_ylim([4,6])
    ax = set_ticks(ax,'y', 4, 6, 3)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=True,
                   left=True, labelleft=False)

    plt.savefig(f'denoising/benchmarks/plots/timings_dn.pdf',
                bbox_inches='tight', dpi=250)
    plt.close()


def testing_plots():
    pass


def main(dirname):
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.titlesize'] = 20
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    training_plots(dirname)

    testing_plots()


if __name__ == '__main__':
    args = vars(PARSER.parse_args())
    start = tm.time()
    main(**args)
    print(f'Program done in {tm.time()-start}')
