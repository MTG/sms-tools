import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))

import dftAnal, dftSynth
import waveIO as WIO
import math

def stft(x, fs, w, N, H):
# Analysis/synthesis of a sound using the short-time fourier transform
# x: input sound, w: analysis window, N: FFT size, H: hop size
# returns y: output sound
  M = w.size                                     # size of analysis window
  hM1 = int(math.floor((M+1)/2))                 # half analysis window size by rounding
  hM2 = int(math.floor(M/2))                     # half analysis window size by floor
  x = np.append(np.zeros(hM2),x)                 # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hM1))                 # add zeros at the end to analyze last sample
  pin = hM1                                      # initialize sound pointer in middle of analysis window       
  pend = x.size-hM1                              # last sample to start a frame
  w = w / sum(w)                                 # normalize analysis window
  y = np.zeros(x.size)                           # initialize output array
  while pin<=pend:                               # while sound pointer is smaller than last sample      
  #-----analysis-----  
    x1 = x[pin-hM1:pin+hM2]                      # select one frame of input sound
    mX, pX = dftAnal.dftAnal(x1, w, N)           # compute dft
  #-----synthesis-----
    y1 = dftSynth.dftSynth(mX, pX, M)            # compute idft
    y[pin-hM1:pin+hM2] += H*y1                   # overlap-add to generate output sound
    pin += H                                     # advance sound pointer
  y = np.delete(y, range(hM2))                   # delete half of first window which was added in stftAnal
  y = np.delete(y, range(y.size-hM1, y.size))    # add zeros at the end to analyze last sample
  return y

def defaultTest():
  str_time = time.time()    
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/orchestra.wav'))
  w = np.hamming(500)
  N = 1024
  H = 125
  y = stft(x, fs, w, N, H)
  print "time taken for computation " + str(time.time()-str_time)  

# example call of stft function
if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/orchestra.wav')
  w = np.hamming(500)
  N = 1024
  H = 100
  y = stft(x, fs, w, N, H)
  WIO.play(y, fs)
  error = -(20*np.log10(2**15) - 20*np.log10(sum(abs(x[N:x.size-N]-y[N:x.size-N]))))
  print "output/input error (in dB) =", error
