import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt
import time as tm

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--dir_name", "-p", default="../datasets/20200901",
                    type=str, help='Directory path to datasets')

tdc_min = 0
tdc_max = 6000
channels = 2560*6

def main(dir_name):
    #f_wire = glob.glob(os.path.join(dirname, 'wire/*'))
    f_simch = glob.glob(os.path.join(dirname, 'simch/*'))
    for f_w,f_s in zip(f_wire, f_simch):
        #wire = np.load(f_w)
        simch = np.load(f_s)

        #ensure energy deposits are inside tdc window
        simch = simch[simch[:,2]<tdc_max]
        simch = simch[simch[:,2]>tdc_min]

        #charge and energy deposits on wires
        ch_depo = np.zeros([channels,tdc_max-tdc_min])
        en_depo = np.zeros_like(ch_depo)

        for i in simch:
            ch_depo[i[1], i[2]] += i[4]
            en_depo[i[1], i[2]] += i[5]

        fname = f_s.split('/')
        fname[-1] = fname[-1].replace('wire', 'charge')
        fname = '/'.join(fname)
        np.save(fname, ch_depo)
        fname = fname.replace('charge', 'energy')
        np.save(fname, en_depo)


if __name__ == '__main__':
    ARGS = vars(PARSER.parse_args())
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
