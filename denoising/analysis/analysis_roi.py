import sys
import os
import argparse
import numpy as np
import time as tm
import matplotlib.pyplot as plt
import matplotlib as mpl

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def training_metrics(warmup):
    dir_name = f'./denoising/output/CNN_{warmup}_final/metrics/'
    dir_name_gc = f'./denoising/output/GCNN_{warmup}_final/metrics/'
    
    fname = dir_name + 'loss_sum.npy'
    loss = np.load(fname)[0]
    # shape (1. train_epochs)

    fname = dir_name + 'test_epochs.npy'
    val_epochs = np.load(fname)
    # shape (val_epochs,)

    fname = dir_name + 'test_metrics.npy'    
    val_metrics = np.load(fname)

    fname = dir_name_gc + 'loss_sum.npy' 
    loss_gc = np.load(fname)[0]

    fname = dir_name_gc + 'test_epochs.npy' 
    val_epochs_gc = np.load(fname)

    fname = dir_name_gc + 'test_metrics.npy' 
    val_metrics_gc = np.load(fname)

    return ([loss, loss_gc],
            [val_epochs, val_epochs_gc],
            [val_metrics, val_metrics_gc])


def training_timings(warmup):
    dir_name = f'./denoising/output/CNN_{warmup}_final/timings/'
    dir_name_gc = f'./denoising/output/GCNN_{warmup}_final/timings/'
    
    fname = dir_name + 'timings_train.npy'
    timings_train = np.load(fname)
    # shape (epochs,)

    fname = dir_name + 'timings_test.npy'
    timings_val = np.load(fname)
    # shape (val_epochs,)

    fname = dir_name_gc + 'timings_train.npy'
    timings_train_gc = np.load(fname)

    fname = dir_name_gc + 'timings_test.npy'
    timings_val_gc = np.load(fname)

    #print('Mean training times cnn: ', timings_train.mean())
    #print('Mean training times gcnn: ', timings_train_gc.mean())
    #print('Mean validation times cnn: ', timings_val.mean())
    #print('Mean validation times gcnn: ', timings_val_gc.mean())

    return ([timings_train, timings_train_gc],
            [timings_val, timings_val_gc])

def set_ticks(ax, axis, start=None, end=None,
              num_maj=None, div=5, d=0):
    """
    Set both major and minor axes ticks in the logarithmical scale
    Parameters:
        ax: matplotlib.axes.Axes object
        axis: 1 y axis, 0 x axis
        start: int, leftmost tick
        end: int, rightmost tick
        num_maj: int, number of major ticks
        div: int, how to partition interval between maj ticks
        d: int, decimal digits axis labels
    """
    #to divide each in 5 parts: num_min = (num_maj - 1)*5 -1
    rng = end - start
    ticks = [i*rng/(num_maj-1) + start for i in range(num_maj)]
    def format_func(x):
        if x == 0:
            return r'$0$'
        return r'$%s$'%format(x, f'.{d}f')
    labels = list(map(format_func, ticks))
    num_min = (num_maj -1)*div + 1 
    ticks_min = [i*rng/(num_min-1) + start for i in range(num_min)]

    if axis == 'x':
        ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
        ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(ticks_min))
        ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
        return ax

    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    ax.yaxis.set_minor_locator(mpl.ticker.FixedLocator(ticks_min))
    ax.yaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    return ax

def special_ticks(ax, start, end, ticks):
    rng = end - start
    ticks = ticks
    labels = list(map(lambda x:r'$%.0f$'%x, ticks))
    ticks_min = [i for i in range(start,end + 1,5)]

    ax.xaxis.set_major_locator(mpl.ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mpl.ticker.FixedFormatter(labels))
    ax.xaxis.set_minor_locator(mpl.ticker.FixedLocator(ticks_min))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    return ax


def training_plots():
    loss, val_epochs, val_metrics = training_metrics('roi')
    
    epochs = [i for i in range(len(loss[0]))]

    fig = plt.figure()
    fig.suptitle('Training Loss')
    
    gs = fig.add_gridspec(nrows=1, ncols=2, wspace=0.2)
    ax = fig.add_subplot(gs[0])
    ax.set_title('Training')
    ax.set_ylabel(r'Binary Cross Entropy')
    ax.plot(epochs, loss[0], label='cnn', color='#ff7f0e')
    ax.plot(epochs, loss[1], label='gcnn', color='b')
    ax.set_xlim([0,100])
    ax = set_ticks(ax,'x', 0, 100, 6)
    ax.set_ylim([0,.7])
    ax = set_ticks(ax,'y', 0, .7, 5, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    ax.legend(frameon=False, loc='upper right', bbox_to_anchor=(0.65,1.02))

    ax = fig.add_subplot(gs[1])
    ax.set_title('Validation')
    ax.errorbar(val_epochs[1], val_metrics[0][0], yerr=val_metrics[0][1],
                label='cnn', linestyle='--', color='#ff7f0e', marker='s')
    ax.errorbar(val_epochs[1], val_metrics[1][0], yerr=val_metrics[1][1],
                label='gcnn', linestyle='--', color='b', marker='^')
    ax.set_xlim([0,100])
    ax = set_ticks(ax,'x', 0, 100, 6)
    ax.set_ylim([0,7])
    ax = set_ticks(ax,'y', 0, 7, 8)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=True,
                   left=True, labelleft=False)

    plt.savefig(f'denoising/benchmarks/plots/training_loss_roi.pdf',
                bbox_inches='tight', dpi=250)
    plt.close()

    ##########################################################################

    timings_train, timings_val = training_timings('roi')

    fig = plt.figure()
    fig.suptitle('Timings')
    gs = fig.add_gridspec(nrows=4, ncols=2, wspace=0.2, hspace=0.2)

    ax = fig.add_subplot(gs[:3,0])
    ax.set_title('Training')
    ax.set_ylabel('Time [s]')
    ax.plot(epochs, timings_train[0]/94, label='cnn', color='#ff7f0e')
    ax.plot(epochs, timings_train[1]/94, label='gcnn', color='b')
    ax.set_xlim([0,100])
    ax = set_ticks(ax,'x', 0, 100, 6)
    ax.set_ylim([0,1])
    ax = set_ticks(ax,'y', 0, 1, 6, div=4 , d=1)
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
    ax.set_xlim([0,100])
    ax = set_ticks(ax,'x', 0, 100, 6)
    ax.set_ylim([2,6])
    ax = set_ticks(ax,'y', 2, 6, 3, div=4)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    ax = fig.add_subplot(gs[:3, 1])
    ax.set_title('Validation')
    ax.plot(val_epochs[0], timings_val[0]/136, label='cnn',
            color='#ff7f0e', linestyle='--')
    ax.plot(val_epochs[1], timings_val[1]/272, label='gcnn',
            color='b', linestyle='--')
    ax.set_xlim([5,100])
    ax = special_ticks(ax, 5, 100, [5,25,50,75,100])
    ax.set_ylim([0,0.5])
    ax = set_ticks(ax,'y', 0, 0.5, 6, div=4, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=True,
                   left=True, labelleft=False)

    ax = fig.add_subplot(gs[3, 1])
    ax.set_xlabel('Epoch')
    ax.plot(val_epochs[0], timings_val[1]/timings_val[0]/2, linestyle='--')
    ax.set_xlim([5,100])
    ax = special_ticks(ax, 5, 100, [5,25,50,75,100])
    ax.set_ylim([6.5,7.5])
    ax = set_ticks(ax,'y', 6.5, 7.5, 3, div=4, d=1)
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=True,
                   left=True, labelleft=False)

    plt.savefig(f'denoising/benchmarks/plots/timings_roi.pdf',
                bbox_inches='tight', dpi=250)
    plt.close()


def testing_res():
    fname = '../datasets/denoising/test/planes/collection_clear.npy'
    y_true = np.load(fname).flatten()

    dir_name = './denoising/output/CNN_dn_final/final_test/'
    fname = dir_name + 'roi_test_res.npy'
    y_pred = np.load(fname).flatten()

    dir_name = './denoising/output/GCNN_dn_final/final_test/'
    fname = dir_name + 'roi_test_res.npy'
    y_pred_gc = np.load(fname).flatten()

    return [y_true, y_pred, y_pred_gc]

def confusion_matrix(hit, no_hit, t):
    """
    Parameters:
        hit: np.array, scores of real hits
        no_hit: np.array, scores of real no-hits
        t: float, threshold
    Returns:
        tp, fp, fn, tn
    """
    tp = np.count_nonzero(hit > t)
    fn = np.size(hit) - tp

    tn = np.count_nonzero(no_hit < t)
    fp = np.size(no_hit) - tn
    
    return tp, fp, fn, tn

def compute_roc(hit, no_hit):
    fpr = []
    tpr = []

    #don't compute t==0 or t==1 which are trivial
    for t in range(19,0,-1):
        tp, fp, fn, tn = confusion_matrix(hit, no_hit, t/20)
        fpr.append(fp/(tn + fp))
        tpr.append(tp/(tp + fn))          

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    fpr = np.concatenate([[0.],fpr,[1.]],0)
    tpr = np.concatenate([[0.],tpr,[1.]],0)

    auc = ((fpr[1:] - fpr[:-1])*tpr[1:]).sum()

    return fpr, tpr, auc


def testing_plots():
    y = testing_res()
    mask = y[0].astype(bool)
    hit = y[1][mask]
    no_hit = y[1][~mask]

    hit_gc = y[2][mask]
    no_hit_gc = y[2][~mask]

    fig = plt.figure()
    fig.suptitle('Final evaluation')   
    ax = fig.add_subplot()
    ax.set_xlabel('NN score')
    ax.hist(hit,100,range=(0,1), histtype='step',
            label='cnn hit', color='r')
    ax.hist(no_hit,100,range=(0,1), histtype='step',
            label='cnn no-hit', color='g')
    ax.hist(hit_gc,100,range=(0,1), histtype='step',
            label='gcnn hit', color='r', linestyle='--')
    ax.hist(no_hit_gc,100,range=(0,1), histtype='step',
            label='gcnn no-hit', color='g', linestyle='--')
    ax.legend(frameon=False, ncol=2)
    ax.set_yscale('log')
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    plt.savefig('denoising/benchmarks/plots/scores_roi_test.pdf',
                bbox='tight',dpi=300)
    plt.close()

    fpr, tpr, auc = compute_roc(hit, no_hit)
    fpr_gc, tpr_gc, auc_gc = compute_roc(hit_gc, no_hit_gc)

    fig = plt.figure()
    fig.suptitle('Final evaluation: ROC curve')
    ax = fig.add_subplot()
    ax.set_xlabel('False Positive ratio')
    ax.set_ylabel('Sensitivity')
    ax.plot(fpr, tpr, label='cnn, AUC=%.4f'%auc, color='#ff7f0e')
    ax.plot(fpr_gc, tpr_gc, label='gcnn, AUC=%.4f'%auc_gc, color='b')
    n = [i/20 for i in range(21)]
    ax.plot(n,n, lw=0.8, linestyle='--', color='grey', alpha=0.4)
    ax.legend(frameon=False)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax = set_ticks(ax,'x', 0, 1, 6, div=4, d=1)
    ax = set_ticks(ax,'y', 0, 1, 6, div=4, d=1)    
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    plt.savefig('denoising/benchmarks/plots/roc_roi_test.pdf',
                bbox='tight',dpi=300)
    plt.close()

    print('AUC cnn', auc)
    print('AUC gcnn', auc_gc)


def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.titlesize'] = 20
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14
    
    training_plots()

    testing_plots()


if __name__ == '__main__':
    start = tm.time()
    main()
    print(f'Program done in {tm.time()-start}')
