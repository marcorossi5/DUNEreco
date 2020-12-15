import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from time import time

def singlenode():
    """
    cnn = [[1,2,3,4],
            [37.78/188, 33.42/94, 26.60/63, 21.62/47],
            [14.01/18, 9.53/12, 7.22/6, 7.12/6]]
    gcnn = [[1,2,3,4],
             [215.23/188, 122.68/94, 84.49/63, 66.53/47],
             [220.52/36, 114.47/18, 85.50/12, 60.17/12]]
    """
    cnn = [[0.5,1.5,2.5,3.5,4.5],
            [37.78, 33.42, 26.60, 21.62],
            [14.01, 9.53, 7.22, 7.12]]
    gcnn = [[0.5,1.5,2.5,3.5,4.5],
             [215.23, 122.68, 84.49, 66.53],
             [220.52, 114.47, 85.50, 60.17]]
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.05)
    fig.suptitle("DataParallel")
    ax = fig.add_subplot(gs[0])
    ax.set_title("Training")
    ax.set_ylabel("cnn")
    ax.hist(cnn[0][:-1], cnn[0], weights=cnn[1], histtype='bar', alpha=0.6, rwidth=0.8, color="darkcyan")
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = fig.add_subplot(gs[1])
    ax.set_xlabel("Num gpus")
    ax.set_ylabel("gcnn")
    ax.hist(gcnn[0][:-1], cnn[0], weights=gcnn[1], histtype='bar', alpha=0.6, rwidth=0.8, color="red")
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    fig.text(0.915, 0.5, r"Time per epoch $[s]$", va='center', rotation=-90, fontsize=15)
    fname = 'denoising/benchmarks/plots/dp_training.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.05)
    fig.suptitle("DataParallel")
    ax = fig.add_subplot(gs[0])
    ax.set_title("Inference")
    ax.set_ylabel("cnn")
    ax.hist(cnn[0][:-1], cnn[0], weights=cnn[2], histtype='bar', alpha=0.6, rwidth=0.8, color="darkcyan")
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)

    ax = fig.add_subplot(gs[1])
    ax.set_xlabel("Num gpus")
    ax.set_ylabel("gcnn")
    ax.hist(gcnn[0][:-1], gcnn[0], weights=gcnn[2], histtype='bar', alpha=0.6, rwidth=0.8, color="red" )
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    fig.text(0.915, 0.5, r"Time per epoch $[s]$", va='center', rotation=-90, fontsize=15)
    fname = 'denoising/benchmarks/plots/dp_testing.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    """
    fig = plt.figure()
    fig.suptitle("DataParallel")
    ax = fig.add_subplot()
    ax.set_title("Training")
    ax.set_xlabel("Num gpus")
    ax.set_ylabel("time per batch")
    ax.plot(cnn[0], cnn[1], label=f"cnn")    
    ax.plot(gcnn[0], gcnn[1], label=f"gcnn")
    ax.legend(frameon=False)
    fname = 'denoising/benchmarks/plots/dp_training.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    fig.suptitle("DataParallel")
    ax = fig.add_subplot()
    ax.set_title("Test")
    ax.set_xlabel("Num gpus")
    ax.set_ylabel("time per batch")
    ax.plot(cnn[0], cnn[2], label=f"cnn")    
    ax.plot(gcnn[0], gcnn[2], label=f"gcnn")
    ax.legend(frameon=False)
    fname = 'denoising/benchmarks/plots/dp_testing.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    """

def multinode():
    """
    cnn = {"1": [[1,2,3,4],
                 [27.91/188, 11.71/94, 10.91/63, 8.21/47],
                 [3.19/18, 2.08/9, 2.16/6, 1.78/5]],
           "2": [[2,4,6,8],
                 [17.8/94, 7.27/47, 6.14/32, 4.53/24],
                 [1.65/9, 1.20/5, 0.95/3, 1.05/3]],
           "3": [[3,6,12],
                 [11.74/63, 5.28/32, 3.23/16],
                 [1.23/6, 0.73/3, 0.59/2]]}
    gcnn = {"1": [[1,2,3,4],
                  [52.52/188, 36.12/94, 43.59/63, 33.26/47],
                  [24.16/69, 22.38/35, 27.72/23, 21.11/18]],
            "2": [[2,4,6,8],
                  [38.47/94, 22.15/47, 25.48/32, 19.23/24],
                  [13.55/35, 11.6/18, 14.45/12, 10.89/9]],
            "3": [[3,6,12],
                  [26.96/63, 14.63/32, 13.07/16],
                  [9.33/23, 7.23/6, 7.06/6]]}
    """
    m = {3: "s", 4: "o", 6: "P", 8: "^", 12: "D"}
    gc = {"1": "lime", "2": "darkcyan", "3": "darkgreen"}
    cc = {"1": "coral", "2": "red", "3": "mediumslateblue"}
    cnn = {"1": [[1,2,3,4],
                 [27.91, 11.71, 10.91, 8.21],
                 [3.19, 2.08, 2.16, 1.78],
                 [4,4,3,4]],
           "2": [[2,4,6,8],
                 [17.8, 7.27, 6.14, 4.53],
                 [1.65, 1.20, 0.95, 1.05],
                 [8,8,6,8]],
           "3": [[3,6,12],
                 [11.74, 5.28, 3.23],
                 [1.23, 0.73, 0.59],
                 [12,12,12]]}
    gcnn = {"1": [[1,2,3,4],
                  [52.52, 36.12, 43.59, 33.26],
                  [24.16, 22.38, 27.72, 21.11],
                  [4,4,3,4]],
            "2": [[2,4,6,8],
                  [38.47, 22.15, 25.48, 19.23],
                  [13.55, 11.6, 14.45, 10.89],
                  [8,8,6,8]],
            "3": [[3,6,12],
                  [26.96, 14.63, 13.07],
                  [9.33, 7.23, 7.06],
                  [12,12,12]]}

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.05)
    fig.suptitle("DistributedDataParallel")
    ax = fig.add_subplot(gs[0])
    ax.set_title("Training")
    ax.set_ylabel("cnn")
    for node, times in cnn.items():
        times = np.array(times)
        ax.plot(times[0], times[1], c=cc[node], lw=0.75, label=f"{node} nodes")
        for proc, train, _, gpus in times.T:
            ax.scatter(proc, train, marker=m[gpus], c=cc[node])
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    first_legend = ax.legend(frameon=False, loc='center right')
    ax.add_artist(first_legend)
    handles = [ax.scatter([],[], label='gpus:', marker='o', c='white')]
    for gpu, marker in m.items():
        handles.append( ax.scatter([],[], label=gpu, marker=marker, c='royalblue') )
    ax.legend(handles=handles, bbox_to_anchor=(0.1, 0.87, 0.85, .102), ncol=6, mode="expand", borderaxespad=0., frameon=False)

    ax = fig.add_subplot(gs[1])
    ax.set_xlabel("Num subprocesses")
    ax.set_ylabel("gcnn")
    for node, times in gcnn.items():
        times = np.array(times)
        ax.plot(times[0], times[1], c=gc[node], lw=0.75, label=f"{node} nodes")
        for proc, train, _, gpus in times.T:
            ax.scatter(proc, train, marker=m[gpus], c=gc[node])
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    ax.legend(frameon=False)
    fig.text(0.915, 0.5, r"Time per epoch $[s]$", va='center', rotation=-90, fontsize=15)
    fname = 'denoising/benchmarks/plots/ddp_training.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.05)
    fig.suptitle("DistributedDataParallel")
    ax = fig.add_subplot(gs[0])
    ax.set_title("Inference")
    ax.set_ylabel("cnn")
    for node, times in cnn.items():
        times = np.array(times)
        ax.plot(times[0], times[2], c=cc[node], lw=0.75, label=f"{node} nodes")
        for proc, _, test, gpus in times.T:
            ax.scatter(proc, test, marker=m[gpus], c=cc[node])
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=False)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    first_legend = ax.legend(frameon=False, loc='center right')
    ax.add_artist(first_legend)
    handles = [ax.scatter([],[], label='gpus:', marker='o', c='white')]
    for gpu, marker in m.items():
        handles.append( ax.scatter([],[], label=gpu, marker=marker, c='royalblue') )
    ax.legend(handles=handles, bbox_to_anchor=(0.1, 0.87, 0.85, .102), ncol=6, mode="expand", borderaxespad=0., frameon=False)

    ax = fig.add_subplot(gs[1])
    ax.set_xlabel("Num subprocesses")
    ax.set_ylabel("gcnn")
    for node, times in gcnn.items():
        times = np.array(times)
        ax.plot(times[0], times[2], c=gc[node], lw=0.75, label=f"{node} nodes")
        for proc, _, test, gpus in times.T:
            ax.scatter(proc, test, marker=m[gpus], c=gc[node])
    ax.tick_params(axis='x', which='both', direction='in',
                   top=True, labeltop=False,
                   bottom=True, labelbottom=True)
    ax.tick_params(axis='y', which='both', direction='in',
                   right=True, labelright=False,
                   left=True, labelleft=True)
    ax.legend(frameon=False)
    fig.text(0.915, 0.5, r"Time per epoch $[s]$", va='center', rotation=-90, fontsize=15)
    fname = 'denoising/benchmarks/plots/ddp_testing.pdf'
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
        

def main():
    mpl.rcParams['text.usetex'] = True
    mpl.rcParams['savefig.format'] = 'pdf'
    mpl.rcParams['figure.titlesize'] = 20
    mpl.rcParams['axes.titlesize'] = 17
    mpl.rcParams['axes.labelsize'] = 14
    mpl.rcParams['ytick.labelsize'] = 14
    mpl.rcParams['xtick.labelsize'] = 14
    mpl.rcParams['legend.fontsize'] = 14

    singlenode()
    multinode()


if __name__ == '__main__':
    start = time()
    main()
    print(f"Program done in {time()-start} s")