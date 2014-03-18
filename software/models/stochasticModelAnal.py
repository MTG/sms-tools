import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, resample
from scipy.fftpack import fft, ifft
import time

import waveIO as WIO

def stochasticModelAnal(x, H, stocf) :
  # x: input array sound, H: hop size, 
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # returns mYst: stochastic envelope
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
    mY = resample(np.maximum(-200, mX), mX.size*stocf)     # decimate the mag spectrum 
    if pin == 0:
      mYst = np.array([mY])
    else:
      mYst = np.vstack((mYst, np.array([mY])))
    pin += H                                               # advance sound pointer
  return mYst

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/ocean.wav'))
  H = 256
  stocf = 0.2
  mYst = stochasticModelAnal(x, H, stocf)
  print "time taken for computation " + str(time.time()-str_time)
    
    
# example call of stochasticModelAnal function
if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/ocean.wav')
  H = 256
  stocf = .2
  mYst = stochasticModelAnal(x, H, stocf)
  numFrames = int(mYst[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(stocf*H)*float(fs)/(stocf*2*H)                       
  plt.pcolormesh(frmTime, binFreq, np.transpose(mYst))
  plt.xlabel('Time(s)')
  plt.ylabel('Frequency(Hz)')
  plt.autoscale(tight=True)
  plt.title('stochastic approximation')
  plt.autoscale(tight=True)
  plt.show()
