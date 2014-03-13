import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import ifft
import math

import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import sineModelAnal as SA
import waveIO as WIO
import errorHandler as EH

try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  EH.printWarning(1)
  

def sineModelSynth(tfreq, tmag, tphase, N, H, fs):
  # Synthesis of a sound using the sinusoidal model
  # tfreq,tmag, tphase: frequencies, magnitudes and phases of sinusoids,
  # N: synthesis FFT size, H: hop size, 
  # returns y: output array sound
  hN = N/2                                                # half of FFT size for synthesis
  l = 0                                                   # frame index
  L = tfreq[:,0].size                                     # number of frames
  nTracks = tfreq[0,:].size                               # number of sinusoidal tracks
  pout = 0                                                # initialize output sound pointer         
  ysize = H*(L+3)                                         # output sound size
  yw = np.zeros(N)                                        # initialize output sound frame
  y = np.zeros(ysize)                                     # initialize output array
  sw = np.zeros(N)                                        # initialize synthesis window
  ow = triang(2*H);                                       # triangular window
  sw[hN-H:hN+H] = ow                                      # add triangular window
  bh = blackmanharris(N)                                  # blackmanharris window
  bh = bh / sum(bh)                                       # normalized blackmanharris window
  sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
  while l<L:                                              # iterate over all frames
    ytloc = N*tfreq[l,:]/fs                               # synthesis peak locs
    ytmag = tmag[l,:]                                     # synthesis peak amplitudes
    ytphase = tphase[l,:]                                 # synthesis residual envelope
    Y = GS.genSpecSines(ytloc, ytmag, ytphase, N)         # generate sines in the spectrum         
    fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
    yw[:hN-1] = fftbuffer[hN+1:]                          # undo zero-phase window
    yw[hN-1:] = fftbuffer[:hN+1] 
    y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
    l += 1                                                # advance frame pointer
    pout += H                                             # advance sound pointer
  return y

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.blackman(2001)
  N = 2048
  H = 500
  t = -90
  minSineDur = .01
  maxnSines = 150
  Ns = 512
  H = Ns/4
  tfreq, tmag, tphase = SA.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur)
  y = sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)
  print "time taken for computation " + str(time.time()-str_time)  
  
# example call of sineModelSynth function
if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/bendir.wav')
  w = np.blackman(2001)
  N = 2048
  H = 500
  t = -90
  minSineDur = .01
  maxnSines = 150
  freqDevOffset = 20
  freqDevSlope = 0.02
  Ns = 512
  H = Ns/4
  tfreq, tmag, tphase = SA.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
  y = sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)
  WIO.play(y, fs)
