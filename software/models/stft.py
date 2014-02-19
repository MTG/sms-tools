import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))

import dftAnal, dftSynth
import waveIO as WIO
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft, ifft
import math

def stft(x, fs, w, N, H):
  ''' Analysis/synthesis of a sound using the short-time fourier transform
  x: input array sound, w: analysis window, N: FFT size, H: hop size
  returns y: output array sound '''

  M = w.size                                              # size of analysis window
  hM1 = int(math.floor((M+1)/2))                          # half analysis window size by rounding
  hM2 = int(math.floor(M/2))                              # half analysis window size by floor
  pin = hM1                                                # initialize sound pointer in middle of analysis window       
  pend = x.size-hM1                                       # last sample to start a frame
  w = w / sum(w)                                          # normalize analysis window
  y = np.zeros(x.size)                                    # initialize output array
  while pin<pend:                                         # while sound pointer is smaller than last sample      
  #-----analysis-----  
    x1 = x[pin-hM1:pin+hM2]                               # select one frame of input sound
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft
  #-----synthesis-----
    y1 = dftSynth.dftSynth(mX, pX, M)                     # compute idft
    y[pin-hM1:pin+hM2] += H*y1                            # overlap-add to generate output sound
    pin += H                                              # advance sound pointer
  return y

def defaultTest():
  str_time = time.time()    
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/oboe-A4.wav'))
  w = np.blackman(511)
  N = 1024
  H = 128
  y = stft(x, fs, w, N, H)
  print "time taken for computation " + str(time.time()-str_time)  

# example call of stft function
if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/oboe-A4.wav')
  w = np.blackman(511)
  N = 1024
  H = 128
  y = stft(x, fs, w, N, H)
  WIO.play(y, fs)
  error = -(20*np.log10(2**15) - 20*np.log10(sum(abs(x[N:x.size-N]-y[N:x.size-N]))))
  print "output/input error (in dB) =", error
