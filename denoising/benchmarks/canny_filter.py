""" This module computes the Canny filter for planes in the test set"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.metrics import confusion_matrix
import time as tm

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--threshold", "-t", default=0.5,
                    type=float, help='Threshold to compute predictions')

def classification_metrics(y_true, y_pred, t):
    """
    Computes accuracy, sensitivity and AUC for given prediction
    Parameters:
        y_true: np.array, target
        y_pred: np.array, predictions
        t: float, threshold
    Returns:
        accuracy, sensitivity, specificity, tpr, fpr, auc
    """
    y_true = y_true.flatten().astype(int)
    y_pred = y_pred.flatten()
    
    cm = confusion_matrix(y_true, (y_pred>t).astype(int))

    acc = (cm[1,1] + cm[0,0]) / cm.sum()
    sns = cm[1,1] / (cm[1,1] + cm[1,0])
    spc = cm[0,0] / (cm[0,1] + cm[0,0])

    tpr = []
    fpr = []

    for i in np.linspace(0,1,10)[-2:0:-1]:
        cm = confusion_matrix(y_true, y_pred>i)
        fpr.append(cm[0,1] / (cm[0,1] + cm[0,0]))
        tpr.append(cm[1,1] / (cm[1,1] + cm[1,0]))            

    fpr = np.array(fpr)
    tpr = np.array(tpr)

    fpr = np.concatenate([[0.],fpr,[1.]],0)
    tpr = np.concatenate([[0.],tpr,[1.]],0)

    auc = ((fpr[1:] - fpr[:-1])*tpr[1:]).sum()

    return acc, sns, spc, tpr, fpr, auc


def plot():
    fig = plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=1, wspace=3)

    ax = fig.add_subplot(gs[0])
    ax.plot(img[10], lw=0.3)
    #z = ax.imshow(img)
    #plt.colorbar(z, ax=ax)
    ax.set_title('Labels')

    ax = fig.add_subplot(gs[1])
    ax.plot(filtered_img[10], lw=0.3)
    #z = ax.imshow(filtered_img)
    #plt.colorbar(z, ax=ax)
    ax.set_title('Predicted')
        
    plt.savefig('wiener_filter.png', dpi=300, bbox_inches='tight')
    plt.close()

def main(threshold):
    # Load target images
    file_name = os.path.join("../datasets/denoising/test",
                             'planes', 'collection_clear.npy') 
    img = np.load(file_name)
    img[img!=0] = 1

    sig = np.count_nonzero(img)
    tot = img.size
    print('Percentage of pixels with signal:', sig/tot)

    # Load noisy images
    file_name = os.path.join("../datasets/denoising/test",
                             'planes', 'collection_noisy.npy') 
    # subtract the pedestal already
    # pedestal = 500
    noisy_img = np.load(file_name) - 500

    # Apply Canny Filter
    filtered_img = []
    acc = []
    sns = []
    spc = []
    #tpr = []
    #fpr = []
    auc = []

    for i,j in zip(img, noisy_img):
        # to be used by canny, first normalize in [0,1]
        norm = (j[0]-j[0].min())/(j[0].max()-j[0].min())
        im = canny(norm).astype(float)
        filtered_img.append(im)
        x = classification_metrics(i[0], im, threshold)
        acc.append(x[0])
        sns.append(x[1])
        spc.append(x[2])
        #tpr.append(x[3])
        #fpr.append(x[4])
        auc.append(x[5])

    filtered_img = np.stack(filtered_img)[:, None]

    acc_mean = np.mean(acc)
    acc_std =  np.std(acc) / np.sqrt(len(acc))

    sns_mean = np.mean(sns)
    sns_std =  np.std(sns) / np.sqrt(len(sns))


    spc_mean = np.mean(spc)
    spc_std =  np.std(spc) / np.sqrt(len(spc))

    auc_mean = np.mean(auc)
    auc_std =  np.std(auc) / np.sqrt(len(auc))

    res = np.array([[acc_mean, acc_std],
                    [sns_mean, sns_std],
                    [spc_mean, spc_std],
                    [auc_mean, auc_std]])
    dir_name = 'denoising/benchmarks/results/'
    fname = dir_name + 'canny_metrics'
    np.save(fname, res)

    fname = dir_name + 'canny_res'
    np.save(fname, filtered_img)


if __name__ == '__main__':
    ARGS = vars(PARSER.parse_args())

    START = tm.time()
    main(**ARGS)
    print('Program done in %f'%(tm.time()-START))
