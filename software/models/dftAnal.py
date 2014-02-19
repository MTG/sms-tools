import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))

import waveIO as wp
from scipy.fftpack import fft, ifft
import math

def dftAnal(x, w, N):
  ''' Analysis of a signal using the discrete fourier transform
  x: input signal, w: analysis window, N: FFT size, 
  returns mX: magnitude spectrum, pX: phase spectrum'''

  hN = N/2                                  # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))       # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))           # half analysis window size by floor
  fftbuffer = np.zeros(N)                   # initialize buffer for FFT
  w = w / sum(w)                            # normalize analysis window
  xw = x*w                                  # window the input sound
  fftbuffer[:hM1] = xw[hM2:]                # zero-phase window in fftbuffer
  fftbuffer[N-hM2:] = xw[:hM2]        
  X = fft(fftbuffer)                       # compute FFT
  mX = 20 * np.log10(abs(X[:hN]))          # magnitude spectrum of positive frequencies in dB     
  pX = np.unwrap(np.angle(X[:hN]))         # unwrapped phase spectrum of positive frequencies
  return mX, pX

# example call of dftAnal function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../sounds/oboe-A4.wav')
  w = np.hamming(511)
  N = 512
  pin = 5000
  hM1 = int(math.floor((w.size+1)/2)) 
  hM2 = int(math.floor(w.size/2))  
  x1 = x[pin-hM1:pin+hM2]
  mX, pX = dftAnal(x1, w, N)

  plt.figure(1)
  plt.subplot(311)
  plt.plot(np.arange(-hM1, hM2), x1)
  plt.axis([-hM1, hM2, min(x1), max(x1)])
  plt.ylabel('amplitude')
  plt.title('input signal: x')

  plt.subplot(3,1,2)
  plt.plot(np.arange(N/2), mX, 'r')
  plt.axis([0,N/2,min(mX),max(mX)])
  plt.title ('magnitude spectrum: mX')
  plt.ylabel('amplitude (dB)')

  plt.subplot(3,1,3)
  plt.plot(np.arange(N/2), pX, 'c')
  plt.axis([0,N/2,min(pX),max(pX)])
  plt.title ('phase spectrum: pX')
  plt.ylabel('phase (radians)')

  plt.show()
