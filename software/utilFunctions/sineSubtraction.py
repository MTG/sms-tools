import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal
import errorHandler as EH

try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  EH.printWarning(1)
  

def sineSubtraction(x, N, H, sfreq, smag, sphase, fs):
  # subtract sinusoids from a sound
  # x: input sound, N: fft-size, H: hop-size
  # sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
  # returns xr: residual sound 
  hN = N/2  
  x = np.append(np.zeros(hN),x)                    # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hN))                    # add zeros at the end to analyze last sample
  bh = blackmanharris(N)                           # synthesis window
  w = bh/ sum(bh)                                  # normalize synthesis window
  sw = np.zeros(N)    
  sw[hN-H:hN+H] = triang(2*H) / w[hN-H:hN+H]
  L = sfreq[:,0].size                              # number of frames   
  xr = np.zeros(x.size)                            # initialize output array
  pin = 0

  for l in range(L):
    xw = x[pin:pin+N]*w                            # window the input sound                               
    X = fft(fftshift(xw))                          # compute FFT 
    Yh = GS.genSpecSines(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N)   # generate spec sines          
    Xr = X-Yh                                      # subtract sines from original spectrum
    xrw = np.real(fftshift(ifft(Xr)))              # inverse FFT
    xr[pin:pin+N] += xrw*sw                        # overlap-add
    pin += H   									   # advance sound pointer
  
  xr = np.delete(xr, range(hN))                    # delete half of first window which was added in stftAnal
  xr = np.delete(xr, range(xr.size-hN, xr.size))   # delete half of last window which was added in stftAnal
  
  return xr