import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, resample
from scipy.fftpack import fft, ifft
import math
import time

import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions_C/'))

import smsWavplayer as wp

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def stochasticModelSpectrogramPlot(x, w, N, H, stocf, maxFreq) :
  # x: input array sound, w: analysis window, N: FFT size, H: hop size, 
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # y: output sound

  hN = N/2                                                 # size of positive spectrum
  hM = (w.size)/2                                          # half analysis window size
  pin = hM                                                 # initialize sound pointer in middle of analysis window       
  pend = x.size-hM                                         # last sample to start a frame
  fftbuffer = np.zeros(N)                                  # initialize buffer for FFT
  yw = np.zeros(w.size)                                    # initialize output sound frame
  y = np.zeros(x.size)                                     # initialize output array
  w = w / sum(w)                                           # normalize analysis window
  ws = hanning(w.size)*2                                   # synthesis window
  numFrames = int(math.floor(pend/float(H)))
  frmNum = 0
  frmTime = []
  lastBin = N*maxFreq/float(fs)
  binFreq = np.arange(lastBin)*float(fs)/N

  while pin<pend:  
    frmTime.append(pin/float(fs))             
  #-----analysis-----             
    xw = x[pin-hM:pin+hM] * w                              # window the input sound
    X = fft(xw)                                            # compute FFT
    mX = 20 * np.log10(abs(X[:hN]))                        # magnitude spectrum of positive frequencies
    mXenv = resample(np.maximum(-200, mX), mX.size*stocf)  # decimate the mag spectrum     
    
  #-----synthesis-----
    mY = resample(mXenv, hN)                               # interpolate to original size

    if frmNum == 0:                                       # Accumulate and store STFT
      XSpec = np.transpose(np.array([mX[:lastBin]]))
      YSpec = np.transpose(np.array([mY[:lastBin]]))
    else:
      XSpec = np.hstack((XSpec,np.transpose(np.array([mX[:lastBin]]))))
      YSpec = np.hstack((YSpec,np.transpose(np.array([mY[:lastBin]]))))
 

    pY = 2*np.pi*np.random.rand(hN)                      # generate phase random values
    Y = np.zeros(N, dtype = complex)
    Y[:hN] = 10**(mY/20) * np.exp(1j*pY)                   # generate positive freq.
    Y[hN+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1])  # generate negative freq.

    fftbuffer = np.real( ifft(Y) )                         # inverse FFT
    y[pin-hM:pin+hM] += H*ws*fftbuffer                     # overlap-add
    pin += H 
    frmNum += 1
  
  frmTime = np.array(frmTime)                               # The time at the centre of the frames
  plt.figure(1)
  plt.subplot(2,1,1)
  plt.pcolormesh(frmTime,binFreq,XSpec)
  plt.autoscale(tight=True)
  plt.title('X spectrogram')

  plt.subplot(2,1,2)
  plt.pcolormesh(frmTime,binFreq,YSpec)
  plt.autoscale(tight=True)
  plt.title('X stochastic approx. spectrogram')
  plt.show()
  
  return y    
    
# example call of stochasticModel function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../sounds/ocean.wav')
  w = np.hamming(1024)
  N = 1024
  H = 512
  stocf = 0.3
  maxFreq = fs
  y = stochasticModelSpectrogramPlot(x, w, N, H, stocf, maxFreq)
  wp.play(y, fs)