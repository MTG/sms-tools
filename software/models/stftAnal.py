import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))

import dftAnal
import smsWavplayer as wp
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft
import math

def stftAnal(x, fs, w, N, H) :
  ''' Analysis of a sound using the short-time fourier transform
  x: input array sound, w: analysis window, N: FFT size, H: hop size
  returns xmX: magnitude spectra, xpX: phase spectra '''
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
    if pin == hM1: 
      xmX = np.array([mX])
      xpX = np.array([pX])
    else:
      xmX = np.vstack((xmX,np.array([mX])))
      xpX = np.vstack((xpX,np.array([pX])))
    pin += H                                              # advance sound pointer
  return xmX, xpX

def defaultTest():
  str_time = time.time()    
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/piano.wav'))
  w = np.hamming(512)
  N = 512
  H = 128
  mX, pX = stftAnal(x, fs, w, N, H)
  print "time taken for computation " + str(time.time()-str_time)  

# example call of stftAnal function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../sounds/piano.wav')
  w = np.hamming(1024)
  N = 1024
  H = 512
  mX, pX = stftAnal(x, fs, w, N, H)
  
  plt.figure(1)
  plt.subplot(211)
  numFrames = int(mX[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(N/2)*float(fs)/N                         
  plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
  plt.xlabel('time (sec)')
  plt.ylabel('frequency (Hz)')
  plt.title('magnitude spectrogram')
  plt.autoscale(tight=True)

  plt.subplot(212)
  numFrames = int(pX[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(N/2)*float(fs)/N                         
  plt.pcolormesh(frmTime, binFreq, np.transpose(np.diff(pX,axis=1)))
  plt.xlabel('time (sec)')
  plt.ylabel('frequency (Hz)')
  plt.title('phase spectrogram (derivative)')
  plt.autoscale(tight=True)
  plt.show()
  
