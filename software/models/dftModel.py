import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))

import waveIO as WIO
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft
import math

def dftModel(x, w, N):
  ''' Analysis/synthesis of a signal using the discrete fourier transform
  x: input signal, w: analysis window, N: FFT size, y: output signal'''

  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  y = np.zeros(x.size)                                    # initialize output array
  #----analysis--------
  xw = x*w                                                # window the input sound
  fftbuffer[:hM1] = xw[hM2:]                              # zero-phase window in fftbuffer
  fftbuffer[N-hM2:] = xw[:hM2]        
  X = fft(fftbuffer)                                      # compute FFT
  mX = 20 * np.log10(abs(X[:hN]))                         # magnitude spectrum of positive frequencies in dB     
  pX = np.unwrap(np.angle(X[:hN]))                        # unwrapped phase spectrum of positive frequencies
  #-----synthesis-----
  Y = np.zeros(N, dtype = complex)                        # clean output spectrum
  Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                    # generate positive frequencies
  Y[hN+1:] = 10**(mX[:0:-1]/20) * np.exp(-1j*pX[:0:-1])   # generate negative frequencies
  fftbuffer = np.real(ifft(Y))                            # compute inverse FFT
  y[:hM2] = fftbuffer[N-hM2:]                             # undo zero-phase window
  y[hM2:] = fftbuffer[:hM1]
  return y

# example call of dftModel function
if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/oboe-A4.wav')
  w = np.blackman(511)
  N = 1024
  pin = 5000
  hM1 = int(math.floor((w.size+1)/2)) 
  hM2 = int(math.floor(w.size/2))  
  x1 = x[pin-hM1:pin+hM2]
  y = dftModel(x1, w, N)

  plt.figure(1)
  plt.subplot(211)
  plt.plot(np.arange(-hM1, hM2), x1)
  plt.axis([-hM1, hM2, min(x1), max(x1)])
  plt.ylabel('amplitude')
  plt.title('input signal')

  plt.subplot(212)
  plt.plot(np.arange(-hM1, hM2), y)
  plt.axis([-hM1, hM2, min(y), max(y)])
  plt.ylabel('amplitude')
  plt.title('output signal')
  plt.show()

  error = -(20*np.log10(2**15) - 20*np.log10(sum(abs(x1*w-y))))
  print "output/input error (in dB) =", error
