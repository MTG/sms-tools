import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
from scipy.interpolate import interp1d
import math
import sys, os, time
import hpsAnal

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as wp
import peakProcessing as PP

try:
  import genSpecSines_C as GS
  import twm_C as fd
except ImportError:
  import genSpecSines as GS
  import twm as fd
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def hpsSynth(hloc, hmag, mXrenv, Ns, H, fs):
  # Synthesis of a sound using the harmonic plus stochastic model
  # hloc: harmonic locations, hmag:harmonic amplitudes, mXrenv: residual envelope
  # Ns: synthesis FFT size, H: hop size, fs: sampling rate 
  # y: output sound, yh: harmonic component, yst: stochastic component
  hNs = Ns/2                                                # half of FFT size for synthesis
  l = 0                                                     # frame index
  L = hloc[:,0].size                                        # number of frames
  nH = hloc[0,:].size                                       # number of harmonics
  pout = 0                                                  # initialize output sound pointer         
  ysize = H*(L+3)                                           # output sound size
  yhw = np.zeros(Ns)                                        # initialize output sound frame
  ysw = np.zeros(Ns)                                        # initialize output sound frame
  yh = np.zeros(ysize)                                      # initialize output array
  ys = np.zeros(ysize)                                      # initialize output array
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                          # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                   # synthesis window
  bh = bh / sum(bh)                                         # normalize synthesis window
  wr = bh                                                   # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]       # synthesis window for harmonic component
  sws = H*hanning(Ns)/2                                     # synthesis window for stochastic component
  lastyhloc = np.zeros(nH)                                  # initialize synthesis harmonic locations
  yhphase = 2*np.pi * np.random.rand(nH)                    # initialize synthesis harmonic phases     
  while l<L:
    yhloc = hloc[l,:]                                       # synthesis harmonics locs
    yhmag = hmag[l,:]                                       # synthesis harmonic amplitudes
    mYrenv = mXrenv[l,:]                                    # synthesis residual envelope
    f0 = yhloc[0]
    yf0 = f0
    yhphase += 2*np.pi * (lastyhloc+yhloc)/2/Ns*H           # propagate phases
    lastyhloc = yhloc 
    Yh = GS.genSpecSines(yhloc, yhmag, yhphase, Ns)         # generate spec sines 
    mYs = resample(mYrenv, hNs)                             # interpolate to original size
    mYs = 10**(mYs/20)                                      # dB to linear magnitude  
    pYs = 2*np.pi * np.random.rand(hNs)                     # generate phase random values
    
    Ys = np.zeros(Ns, dtype = complex)
    Ys[:hNs] = mYs * np.exp(1j*pYs)                         # generate positive freq.
    Ys[hNs+1:] = mYs[:0:-1] * np.exp(-1j*pYs[:0:-1])        # generate negative freq.

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Yh))                          # inverse FFT of harm spectrum                        
    yhw[:hNs-1] = fftbuffer[hNs+1:]                         # undo zer-phase window
    yhw[hNs-1:] = fftbuffer[:hNs+1] 

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Ys))                           # inverse FFT of stochastic approximation spectrum
    ysw[:hNs-1] = fftbuffer[hNs+1:]                         # undo zero-phase window
    ysw[hNs-1:] = fftbuffer[:hNs+1]
    yh[pout:pout+Ns] += sw*yhw                              # overlap-add for sines
    ys[pout:pout+Ns] += sws*ysw                             # overlap-add for stoch
    l += 1                                                  # advance frame pointer
    pout += H                                               # advance sound pointer
  y = yh+ys                                                 # sum harmonic and stochastic components
  return y, yh, ys

if __name__ == '__main__':
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(801)
  N = 1024
  t = -90
  nH = 50
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxhd = 0.2
  stocf = 0.2
  maxnpeaksTwm = 5
  hloc, hmag, mXrenv, Ns, H = hpsAnal.hpsAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
  y, yh, yst = hpsSynth(hloc, hmag, mXrenv, Ns, H, fs)
  wp.play(y, fs)
  wp.play(yh, fs)
  wp.play(yst, fs)


