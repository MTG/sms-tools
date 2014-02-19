import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import math

import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import sineModelAnal
import waveIO as wp
import errorHandler as EH

try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  EH.printWarning(1)
  

def sineModelSynth(ploc, pmag, pphase, N, Ns, H):
  # Synthesis of a sound using the sinusoidal model
  # ploc: peak locations, pmag: peak magnitudes, pphase: peak phases, N: analysis FFT size,
  # Ns: synthesis FFT size, H: hop size, 
  # returns y: output array sound
  hNs = Ns/2                                              # half of FFT size for synthesis
  l = 0                                                   # frame index
  L = ploc[:,0].size                                      # number of frames
  nPeaks = ploc[0,:].size                                 # number of peaks
  pout = 0                                                # initialize output sound pointer         
  ysize = H*(L+3)                                         # output sound size
  yw = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(ysize)                                     # initialize output array
  sw = np.zeros(Ns)                                       # initialize synthesis window
  ow = triang(2*H);                                       # triangular window
  sw[hNs-H:hNs+H] = ow                                    # add triangular window
  bh = blackmanharris(Ns)                                 # blackmanharris window
  bh = bh / sum(bh)                                       # normalized blackmanharris window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H]/bh[hNs-H:hNs+H]       # normalized synthesis window
  while l<L:                                              # iterate over all frames
    yploc = ploc[l,:]*Ns/N                                # synthesis peak locs
    ypmag = pmag[l,:]                                     # synthesis peak amplitudes
    ypphase = pphase[l,:]                                 # synthesis residual envelope
    Y = GS.genSpecSines(yploc, ypmag, ypphase, Ns)        # generate sines in the spectrum         
    fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
    yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yw[hNs-1:] = fftbuffer[:hNs+1] 
    y[pout:pout+Ns] += sw*yw                              # overlap-add and apply a synthesis window
    l += 1                                                # advance frame pointer
    pout += H                                             # advance sound pointer
  return y

def defaultTest():
  str_time = time.time()
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.hamming(511)
  N = 1024
  t = -60
  Ns = 512
  H = Ns/4
  ploc, pmag, pphase = sineModelAnal.sineModelAnal(x, fs, w, N, H, t)
  y = sineModelSynth(ploc, pmag, pphase, N, Ns, H)
  print "time taken for computation " + str(time.time()-str_time)  
  
# example call of sineModel function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../sounds/bendir.wav')
  w = np.hamming(1001)
  N = 2048
  t = -80
  Ns = 512
  H = Ns/4
  ploc, pmag, pphase = sineModelAnal.sineModelAnal(x, fs, w, N, H, t)
  y = sineModelSynth(ploc, pmag, pphase, N, Ns, H)
  wp.play(y, fs)
