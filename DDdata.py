#Used to deoppler high time resolution data

from bifrost.fdmt import Fdmt
import numpy as np
import pylab as plt
import bifrost as bf
from astropy import units as u
import matplotlib.pyplot as plt
import hyperseti
import blimpy as bl

fdmt = Fdmt()

def ddData(max_delay,gulp_size,fstart,fstop,tstart,tstop):
    #fp is file path, for high time resoltuion data
    fp = '/group/director2183/schopra/python_files-from_jupyter/cho134/cho134/data/Voyager1.single_coarse.fine_res.h5'

    fil = bl.Waterfall(fp, f_start=fstart, f_stop=fstop, t_start=tstart, t_stop=tstop)
    fil.data = fil.data[:, 0:1, :]

    max_delay = max_delay
    gulp_size = gulp_size


    def get_gulp(idx):
        d_cpu = fil.data
        d_cpu = np.expand_dims(np.ascontiguousarray(np.swapaxes(d_cpu,0,1).squeeze()),0)[..., idx * gulp_size:(idx+1) * gulp_size]
        return d_cpu
    d_cpu = get_gulp(0)
    print(d_cpu.shape)

    ## Initialize FDMT
    n_disp = max_delay
    n_time = d_cpu.shape[2]
    n_chan = d_cpu.shape[1]

    fdmt.init(n_chan, n_disp, fil.header['fch1'], d_cpu.shape[0], space="cuda")

    # Input shape is (1, n_freq, n_time)
    d_in = bf.ndarray(d_cpu, dtype='f32', space='cuda')
    d_out = bf.ndarray(np.zeros(shape=(1, n_disp, n_time)), dtype='f32', space='cuda')


    # Execute FDMT
    fdmt.execute(d_in, d_out, negative_delays=True)

    d_out = d_out.copy(space='system')

    plt.imshow(np.log(np.array(d_out)).squeeze(), aspect='auto')
    plt.savefig("ddData.png")

# example run
ddData(128,4*8192,8419.296,8419.298,0,300)
