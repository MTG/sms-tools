import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))

import stftAnal, dftSynth
import smsWavplayer as wp
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft
import math

def stftSynth(mY, pY, M, H) :
  ''' Synthesis of a sound using the short-time fourier transform
  mY: magnitude spectra, pY: phase spectra, M: window size, H: hop-size
  returns y: output sound '''
  hM1 = int(math.floor((M+1)/2))                          # half analysis window size by rounding
  hM2 = int(math.floor(M/2))                              # half analysis window size by floor
  nFrames = int(mY[1,:].size)                             # number of frames
  y = np.zeros(nFrames*H + hM1 + hM2)                     # initialize output array
  pin = hM1                  
  for i in range(nFrames):                                # iterate over all frames      
    y1 = dftSynth.dftSynth(mY[:,i], pY[:,i], M)                     # compute idft
    y[pin-hM1:pin+hM2] += H*y1                            # overlap-add to generate output sound
    pin += H                                              # advance sound pointer
  return y

def defaultTest():
  str_time = time.time()    
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/piano.wav'))
  w = np.hamming(1024)
  N = 1024
  H = 512
  mX, pX = stftAnal(x, fs, w, N, H)
  print "time taken for computation " + str(time.time()-str_time)  

# example call of stftSynth function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../sounds/piano.wav')
  w = np.hamming(1024)
  N = 1024
  H = 512
  mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
  y = stftSynth(mX, pX, w.size, H)
  wp.play(y, fs)   
  
  
