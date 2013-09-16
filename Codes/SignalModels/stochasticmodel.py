import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import hamming, hanning, resample
from scipy.fftpack import fft, ifft
import time

import sys, os

sys.path.append(os.path.realpath('../UtilityFunctions/'))
sys.path.append(os.path.realpath('../UtilityFunctions_C/'))
import f0detectiontwm as fd
import wavplayer as wp
import PeakProcessing as PP

try:
  import UtilityFunctions_C as GS
except ImportError:
  import GenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def stochastic_model(x, w, N, H, stocf) :
  # x: input array sound, w: analysis window, N: FFT size, H: hop size, 
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # y: output sound

  hN = N/2                                                 # size of positive spectrum
  hM = (w.size)/2                                          # half analysis window size
  pin = hM                                                 # initialize sound pointer in middle of analysis window       
  pend = x.size-hM                                         # last sample to start a frame
  fftbuffer = np.zeros(N)                                  # initialize buffer for FFT
  x = np.float32(x) / (2**15)                              # normalize input signal (so 0 dB is the max amp. value)
  yw = np.zeros(w.size)                                    # initialize output sound frame
  y = np.zeros(x.size)                                     # initialize output array
  w = w / sum(w)                                           # normalize analysis window
  ws = hanning(w.size)*2                                   # synthesis window

  while pin<pend:       
            
  #-----analysis-----             
    xw = x[pin-hM:pin+hM] * w                              # window the input sound
    X = fft(xw)                                            # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                      # magnitude spectrum of positive frequencies
    mXenv = resample(np.maximum(-200, mX), mX.size*stocf)  # decimate the mag spectrum     

  #-----synthesis-----
    mY = resample(mXenv, hN)                               # interpolate to original size
    pY = 2*np.pi*np.random.rand(hN)                      # generate phase random values
    # pY = np.zeros(hN)
    Y = np.zeros(N, dtype = complex)
    Y[:hN] = 10**(mY/20) * np.exp(1j*pY)                   # generate positive freq.
    Y[hN+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1])  # generate negative freq.

    fftbuffer = np.real( ifft(Y) )                         # inverse FFT
    y[pin-hM:pin+hM] += H*ws*fftbuffer                     # overlap-add
    pin += H                                               # advance sound pointer
  
  return y

def DefaultTest():
    
    str_time = time.time()
    
    (fs, x) = read('../../sounds/speech-female.wav')
    w = np.hamming(512)
    N = 512
    H = 256
    stocf = 0.5
    y = stochastic_model(x, w, N, H, stocf)

    y *= 2**15
    y = y.astype(np.int16)
    
    print "time taken for computation " + str(time.time()-str_time)
    
    
if __name__ == '__main__':

    (fs, x) = read('../../sounds/speech-female.wav')
    # wp.play(x, fs)
    # fig = plt.figure()
    w = np.hamming(512)
    N = 512
    H = 256
    stocf = 0.5
    y = stochastic_model(x, w, N, H, stocf)

    y *= 2**15
    y = y.astype(np.int16)
    wp.play(y, fs)