import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))

import stftAnal as ST
import waveIO as WIO
import sineModelAnal as HA
import sineSubtraction as SS
  

if __name__ == '__main__':
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.hamming(2001)
  N = 2048
  H = 200
  t = -100
  minSineDur = .02
  maxnSines = 200
  freqDevOffset = 10
  freqDevSlope = 0.001
  mX, pX = ST.stftAnal(x, fs, w, N, H)
  tfreq, tmag, tphase = HA.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
  xr = SS.sineSubtraction(x, N, H, tfreq, tmag, tphase, fs)
  mXr, pXr = ST.stftAnal(xr, fs, hamming(H*2), H*2, H)

  numFrames = int(mXr[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(H)*float(fs)/(H*2)                       
  plt.pcolormesh(frmTime, binFreq, np.transpose(mXr))
  plt.autoscale(tight=True)

  tfreq[tfreq==0] = np.nan
  numFrames = int(tfreq[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs) 
  plt.plot(frmTime, tfreq, color='k', ms=3, alpha=1)
  plt.xlabel('Time(s)')
  plt.ylabel('Frequency(Hz)')
  plt.autoscale(tight=True)
  plt.title('sinusoidal + residual components')
  plt.show()
