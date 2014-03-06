import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, resample
from scipy.fftpack import fft, ifft
import time
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as WIO
import errorHandler as EH
try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  EH.printWarning(1)
  

def stochasticModelAnal(x, w, N, H, stocf) :
  # x: input array sound, w: analysis window, N: FFT size, H: hop size, 
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # returns xmXenv: stochastic envelope
  hN = N/2                                                 # size of positive spectrum
  hM = (w.size)/2                                          # half analysis window size
  pin = hM                                                 # initialize sound pointer in middle of analysis window       
  pend = x.size-hM                                         # last sample to start a frame
  fftbuffer = np.zeros(N)                                  # initialize buffer for FFT
  w = w / sum(w)                                           # normalize analysis window
  while pin<pend:              
  #-----analysis-----             
    xw = x[pin-hM:pin+hM] * w                              # window the input sound
    X = fft(xw)                                            # compute FFT
    mX = 20 * np.log10(abs(X[:hN]))                        # magnitude spectrum of positive frequencies
    mXenv = resample(np.maximum(-200, mX), mX.size*stocf)  # decimate the mag spectrum     
    if pin == hM:
      xmXenv = np.array([mXenv])
    else:
      xmXenv = np.vstack((xmXenv, np.array([mXenv])))
    pin += H                                               # advance sound pointer
  return xmXenv

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/ocean.wav'))
  w = np.hamming(512)
  N = 512
  H = 256
  stocf = 0.2
  mXenv = stochasticModelAnal(x, w, N, H, stocf)
  print "time taken for computation " + str(time.time()-str_time)
    
    
# example call of stochasticModelAnal function
if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/ocean.wav')
  w = np.hamming(512)
  N = 512
  H = 256
  stocf = .2
  mXenv = stochasticModelAnal(x, w, N, H, stocf)
  numFrames = int(mXenv[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(stocf*N/2)*float(fs)/(stocf*N)                       
  plt.pcolormesh(frmTime, binFreq, np.transpose(mXenv))
  plt.xlabel('Time(s)')
  plt.ylabel('Frequency(Hz)')
  plt.autoscale(tight=True)
  plt.title('stochastic approximation')
  plt.autoscale(tight=True)
  plt.show()
