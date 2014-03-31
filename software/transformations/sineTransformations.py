import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
from scipy.interpolate import interp1d

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import sineModel as SM 
import utilFunctions as UF

def sineTimeScaling(sfreq, smag, timeScaling):
  # time scaling of sines
  # sfreq, smag: frequencies and magnitudes of input sines
  # timeScaling: scaling factors, in time-value pairs
  # returns ysfreq, ysmag: frequencies and magnitudes of output sines
  L = sfreq[:,0].size                                         # number of input frames
  outL = int(L*timeScaling[-1]/timeScaling[-2])               # number of synthesis frames
  timeScalingEnv = interp1d(timeScaling[::2]/timeScaling[-2], timeScaling[1::2]/timeScaling[-1])
  ysfreq = sfreq[0,:]                                         # initialize output frame
  ysmag = smag[0,:]                                           # initialize output frame
  indexes = (L-1)*timeScalingEnv(np.arange(outL)/float(outL))
  for l in indexes[1:]:
    ysfreq = np.vstack((ysfreq, sfreq[round(l),:]))
    ysmag = np.vstack((ysmag, smag[round(l),:])) 
  return ysfreq, ysmag

def sineFreqScaling(sfreq, freqScaling):
  # frequency scaling of sinusoids
  # sfreq: frequencies of input sinusoids
  # freqScaling: scaling factors, in time-value pairs
  # returns sfreq: frequencies output sinusoids
  L = sfreq[:,0].size            # number of frames
  freqScaling = np.interp(np.arange(L), L*freqScaling[::2]/freqScaling[-2], freqScaling[1::2]) 
  ysfreq = np.empty_like(sfreq)  # create empty output matrix
  for l in range(L):             # go through all frames
    ysfreq[l,:] = sfreq[l,:] * freqScaling[l]
  return ysfreq

if __name__ == '__main__':
  (fs, x) = UF.wavread('../../sounds/mridangam.wav')
  w = np.hamming(801)
  N = 2048
  t = -90
  minSineDur = .005
  maxnSines = 150
  freqDevOffset = 20
  freqDevSlope = 0.02
  Ns = 512
  H = Ns/4
  tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
  freqScaling = np.array([0, 2.0, 1, .3])           
  ytfreq = sineFreqScaling(tfreq, freqScaling)
  timeScale = np.array([0, 0, .091, .091, .405, .405+.4, .747, .747+.4, .934, .934+.4, 1.259, 1.259+.8, 1.568, 1.568+.8, 1.761, 1.761+1.2, 2.057, 2.057+1.2])          
  ytfreq, ytmag = sineTimeScaling(ytfreq, tmag, timeScale)
  y = SM.sineModelSynth(ytfreq, ytmag, np.array([]), Ns, H, fs)
  UF.play(y, fs)


