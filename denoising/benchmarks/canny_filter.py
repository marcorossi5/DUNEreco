import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from sklearn.metrics import confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from denoising.model_utils import


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

    acc = (cm[1,1] + cm[0,0]) / cm.sum()
    sns = cm[1,1] / (cm[1,1] + cm[1,0])
    spc = cm[0,0] / (cm[0,1] + cm[0,0])

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

def main():
    # Load target images
    file_name = os.path.join("../datasets/denoising/test",
                             'planes', 'collection_clear.npy') 
    img = np.load(file_name)

    # Load noisy images
    file_name = os.path.join("../datasets/denoising/test",
                             'planes', 'collection_noisy.npy') 
    # subtract the pedestal already
    # pedestal = 500
    noisy_img = np.load(file_name) - 500

    # Apply Canny Filter

    filtered_img = []
    for i in noisy_img:
        im = canny(i[0]).astype(float)
        filtered_img.append(im)
    filtered_img = np.stack(filtered_img)[:, None]

    acc = []
    sns = []
    spc = []
    #tpr = []
    #fpr = []
    auc = []

    for i,j in zip(img, filtered_img):
        x = classification_metrics(i[0], j[0])
        acc.append(x[0])
        sns.append(x[1])
        spc.append(x[2])
        #tpr.append(x[3])
        #fpr.append(x[4])
        auc.append(x[5])

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
    fname = f'denoising/benchmarks/results/canny_metrics'
    np.save(fname, res)

    # must compute the metrics: accuracy, sensitivity, AUC
    # pass only information relative to the single plane,
    # aggregate and then compute mean and std
    # even if canny is not a statistic algorithm it can perform
    # better or worse given the plane


if __name__ == '__main__':
    START = tm.time()
    main()
    print('Program done in %f'%(tm.time()-START))
