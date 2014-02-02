import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, resample
from scipy.fftpack import fft, ifft
import time

import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../code/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../code/basicFunctions_C/'))

import smsWavplayer as wp

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def stochasticModel(x, w, N, stocf) :
  # x: input array sound, w: analysis window, N: FFT size,  
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # y: output sound

  hN = N/2                                                 # size of positive spectrum
  hM = (w.size)/2                                          # half analysis window size
  pin = hM                                                 # initialize sound pointer in middle of analysis window       
  fftbuffer = np.zeros(N)                                  # initialize buffer for FFT
  yw = np.zeros(w.size)                                    # initialize output sound frame
  w = w / sum(w)                                           # normalize analysis window
  ws = hanning(w.size)*2                                   # synthesis window
             
  #-----analysis-----             
  xw = x[pin-hM:pin+hM] * w                              # window the input sound
  X = fft(xw)                                            # compute FFT
  mX = 20 * np.log10( abs(X[:hN]) )                      # magnitude spectrum of positive frequencies
  mXenv = resample(np.maximum(-200, mX), mX.size*stocf)  # decimate the mag spectrum     
  pX = np.angle(X[:hN])
  #-----synthesis-----
  mY = resample(mXenv, hN)                               # interpolate to original size
  pY = 2*np.pi*np.random.rand(hN)                      # generate phase random values
  Y = np.zeros(N, dtype = complex)
  Y[:hN] = 10**(mY/20) * np.exp(1j*pY)                   # generate positive freq.
  Y[hN+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1])  # generate negative freq.

  fftbuffer = np.real( ifft(Y) )                         # inverse FFT
  y = ws*fftbuffer*N/2                     # overlap-add
  
  return mX, pX, mY, pY, y
    
    
# example call of stochasticModel function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../../../sounds/ocean.wav')
  w = np.hanning(1024)
  N = 1024
  stocf = 0.1
  maxFreq = 10000.0
  lastbin = N*maxFreq/fs
  first = 1000
  last = first+w.size
  mX, pX, mY, pY, y = stochasticModel(x[first:last], w, N, stocf)
  
  plt.figure(1)
  plt.subplot(3,1,1)
  plt.plot(np.arange(0, fs/2.0, fs/float(N)), mY, 'b')
  plt.axis([0, maxFreq, -80, max(mX)+3])
  plt.title('mY')
  plt.subplot(3,1,2)
  plt.plot(np.arange(0, fs/2.0, fs/float(N)), pY-np.pi, 'b')
  plt.axis([0, maxFreq, -np.pi, np.pi]) 
  plt.title('pY')
  plt.subplot(3,1,3)
  plt.plot(np.arange(first, last)/float(fs), y)
  plt.axis([first/float(fs), last/float(fs), min(y), max(y)])
  plt.title('yw')
  plt.show()
