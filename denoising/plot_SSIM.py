import os
import ssim
import numpy as np
import torch
import matplotlib.pyplot as plt

from args import Args

def main(args):
	"""Main function: plots SSIM of a batch of crops to select k1, k2 parameters"""
	fname = 



if __name__ == '__main__':
	ARGS = Args(**ARGS)
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
