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
  

def stochasticModel(x, H, stocf) :
  # x: input array sound, H: hop size, 
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # returns y: output sound
  N = H*2                                                  # FFT size                                             # size of positive spectrum
  w = hanning (N)
  x = np.append(np.zeros(H),x)                             # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(H))                             # add zeros at the end to analyze last sample
  pin = 0                                                  # initialize sound pointer in middle of analysis window       
  pend = x.size-N                                          # last sample to start a frame
  y = np.zeros(x.size)                                     # initialize output array
  w = w / sum(w)                                           # normalize analysis window
  ws = hanning(w.size)*2                                   # synthesis window
  while pin<=pend:              
  #-----analysis-----             
    xw = x[pin:pin+N] * w                                  # window the input sound
    X = fft(xw)                                            # compute FFT
    mX = 20 * np.log10(abs(X[:H]))                         # magnitude spectrum of positive frequencies
    mYst = resample(np.maximum(-200, mX), mX.size*stocf)  # decimate the mag spectrum     
  #-----synthesis-----
    mY = resample(mYst, H)                                 # interpolate to original size
    pY = 2*np.pi*np.random.rand(H)                         # generate phase random values
    Y = np.zeros(N, dtype = complex)
    Y[:H] = 10**(mY/20) * np.exp(1j*pY)                    # generate positive freq.
    Y[H+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1])   # generate negative freq.
    fftbuffer = np.real(ifft(Y))                           # inverse FFT
    y[pin:pin+N] += H*ws*fftbuffer                         # overlap-add
    pin += H  
  y = np.delete(y, range(H))                            # delete half of first window which was added 
  y = np.delete(y, range(y.size-H, y.size))             # delete half of first window which was added                                            # advance sound pointer
  return y

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/ocean.wav'))
  H = 128
  stocf = 0.2
  y = stochasticModel(x, H, stocf)
  print "time taken for computation " + str(time.time()-str_time)
    
    
# example call of stochasticModel function
if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/ocean.wav')
  H = 256
  stocf = .2
  y = stochasticModel(x, H, stocf)
  WIO.play(y, fs)
