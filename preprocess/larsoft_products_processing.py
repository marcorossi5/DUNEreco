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

def process_depo(dir_name):
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

def main(dir_name):
    process_depo(dirname)

    #check if things were done right
    f_raw = glob.glob(os.path.join(dirname, 'raw/raw*'))[0]
    f_ch = glob.glob(os.path.join(dirname, 'simch/charge*'))[0]

    raw = np.load(f_raw)[:,2:]
    ch = np.load(f_ch)[:,2:]

    plt.figure()
    gs = fig.add_gridspec(nrows=2, ncols=3)
    ax = plt.add_subplot(gs[0,:-1])
    ax.imshow(ch[1600:2560])
    ax.colorbar()
    ax.set_title('Charge Deposition')

    ax = plt.subplot(gs[1:,-1])
    ax.imshow(raw[1600:2560])
    ax.colorbar()
    ax.set_title('Raw Digits')

    ax = plt.subplot(gs[0,-1])
    ax.plot(ch[2500], lw=0.2)
    ax.set_title('Wire 2500, Energy')

    ax = plt.subplot(gs[1,-1])
    ax.plot(raw[2500], lw=0.2)
    ax.set_title('Raw Wire 2500, Raw')

    plt.savefig('../collection_t.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    ARGS = vars(PARSER.parse_args())
    START = tm.time()
    main(ARGS)
    print('Program done in %f'%(tm.time()-START))
