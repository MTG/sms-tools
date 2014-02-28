import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import math
import sys, os, functools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as WIO
import peakProcessing as PP
import harmonicDetection as HD
import errorHandler as EH
import stftAnal, dftAnal

try:
  import genSpecSines_C as GS
  import twm_C as TWM
except ImportError:
  import genSpecSines as GS
  import twm as TWM
  EH.printWarning(1)

def f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et, maxnpeaksTwm=10):
  # fundamental frequency detection using twm algorithm
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # minf0: minimum f0 frequency in Hz, maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # returns f0
  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))       # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))              # half analysis window size by floor
  pin = hM1                                               # init sound pointer in middle of anal window          
  pend = x.size - hM1                               # last sample to start a frame
  fftbuffer = np.zeros(N)                            # initialize buffer for FFT
  w = w / sum(w)                                       # normalize analysis window
  f0 = []
  while pin<pend:             
    x1 = x[pin-hM1:pin+hM2]                    # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)       # compute dft           
    ploc = PP.peakDetection(mX, hN, t)      # detect peak locations   
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0t = TWM.f0DetectionTwm(ipfreq, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
    f0 = np.append(f0, f0t)
    pin += H                                              # advance sound pointer
  return f0

if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/vignesh.wav')
  w = np.blackman(1201)
  N = 2048
  t = -90
  minf0 = 130
  maxf0 = 300
  f0et = 7
  maxnpeaksTwm = 4
  H = 128

  mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
  f0 = f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et, maxnpeaksTwm)
  maxplotfreq = 2000.0
  numFrames = int(mX[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
  plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
  plt.autoscale(tight=True)
  
  plt.plot(frmTime, f0, linewidth=2, color='k')
  plt.autoscale(tight=True)
  plt.title('f0 on spectrogram')
  plt.show()

