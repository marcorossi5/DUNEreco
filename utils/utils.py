import os
import numpy as np
from torch import nn

def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2])
    				   for x in open('tmp', 'r').readlines()
    				   ]
    return np.argmax(memory_available)

def get_freer_gpus(n):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2])
    				   for x in open('tmp', 'r').readlines()
    				   ]
    ind = np.argsort(memory_available)
    return np.argmax(memory_available)[-n:]

def compute_psnr(image, noised):
    """
    Alert: only from images with max value = 1
    """
    mse = nn.MSELoss()(image, noised).item()
    m = image.max().item()

    if mse == 0:
        return 0
    return 10 * np.log10(m/mse)