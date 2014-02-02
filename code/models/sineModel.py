import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import math


import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions_C/'))

import dftAnal
import smsF0DetectionTwm as fd
import smsWavplayer as wp
import smsPeakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def sineModel(x, fs, w, N, t):
  # Analysis/synthesis of a sound using the sinusoidal model
  # x: input array sound, w: analysis window, N: size of complex spectrum,
  # t: threshold in negative dB 
  # returns y: output array sound

  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  Ns = 512                                                # FFT size for synthesis (even)
  H = Ns/4                                                # Hop size used for analysis and synthesis
  hNs = Ns/2                                              # half of synthesis FFT size
  pin = max(hNs, hM1)                                     # initialize sound pointer in middle of analysis window       
  pend = x.size - max(hNs, hM1)                           # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yw = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window
  sw = np.zeros(Ns)                                       # initialize synthesis window
  ow = triang(2*H);                                       # triangular window
  sw[hNs-H:hNs+H] = ow                                    # add triangular window
  bh = blackmanharris(Ns)                                 # blackmanharris window
  bh = bh / sum(bh)                                       # normalized blackmanharris window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
  while pin<pend:                                         # while input sound pointer is within sound 
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft
    ploc = PP.peakDetection(mX, hN, t)                    # detect locations of peaks
    pmag = mX[ploc]                                       # get the magnitude of the peaks
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
  #-----synthesis-----
    plocs = iploc*Ns/N;                                   # adapt peak locations to size of synthesis FFT
    Y = GS.genSpecSines(plocs, ipmag, ipphase, Ns)        # generate sines in the spectrum         
    fftbuffer = np.real( ifft(Y) )                        # compute inverse FFT
    yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yw[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
    pin += H                                              # advance sound pointer
  return y

def defaultTest():
  str_time = time.time()
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/oboe-A4.wav'))
  w = np.hamming(511)
  N = 1024
  t = -60
  y = sineModel(x, fs, w, N, t)
  print "time taken for computation " + str(time.time()-str_time)  
  
# example call of sineModel function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../sounds/oboe-A4.wav')
  w = np.hamming(511)
  N = 1024
  t = -60
  y = sineModel(x, fs, w, N, t)
  wp.play(y, fs)