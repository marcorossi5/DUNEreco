""" This module computes the Canny filter for planes in the test set"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
#from sklearn.metrics import confusion_matrix
import time as tm


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.analysis_roi import confusion_matrix

#PARSER = argparse.ArgumentParser()
#PARSER.add_argument("--threshold", "-t", default=0.5,
#                    type=float, help='Threshold to compute predictions')


def main():
    # Load target images
    file_name = os.path.join("../datasets/denoising/test",
                             'planes', 'collection_clear.npy') 
    img = np.load(file_name)
    img[img!=0] = 1
    img = img.astype(bool)

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

    for i,j in zip(img, noisy_img):
        # to be used by canny, first normalize in [0,1]
        norm = (j[0]-j[0].min())/(j[0].max()-j[0].min())
        im = canny(norm).astype(int)

        hit = im[i[0]]
        no_hit = im[~i[0]]

        tp, fp, fn, tn = confusion_matrix(hit, no_hit)
        acc.append( (tp+tn)/(tp+fp+fp+fn) )
        sns.append( tp )
        spc.append( tn/(fp+tn) )

        filtered_img.append(im)

    filtered_img = np.stack(filtered_img)[:, None]

    acc_mean = np.mean(acc)
    acc_std =  np.std(acc) / np.sqrt(len(acc))

    sns_mean = np.mean(sns)
    sns_std =  np.std(sns) / np.sqrt(len(sns))

    spc_mean = np.mean(spc)
    spc_std =  np.std(spc) / np.sqrt(len(spc))

    res = np.array([[acc_mean, acc_std],
                    [sns_mean, sns_std],
                    [spc_mean, spc_std]])
    dir_name = 'denoising/benchmarks/results/'
    fname = dir_name + 'canny_metrics'
    np.save(fname, res)

    fname = dir_name + 'canny_res'
    np.save(fname, filtered_img)


if __name__ == '__main__':
    #ARGS = vars(PARSER.parse_args())

    START = tm.time()
    main()
    print('Program done in %f'%(tm.time()-START))
