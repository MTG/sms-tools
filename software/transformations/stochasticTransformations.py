import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
from scipy.interpolate import interp1d

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import stochasticModel as STC
import utilFunctions as UF

def stochasticTimeScale(stocEnv, inTime, timeScaling):
  # time scaling of stochastic component
  # stocEnv: stochastic envelope
  # timeScaling: scaling factors, in time-value pairs
  # returns ystocEnv: stochastic envelope
  L = stocEnv[:,0].size                                       # number of input frames
  outL = int(L*timeScaling[-1]/timeScaling[-2])               # number of synthesis frames
  timeScalingEnv = interp1d(timeScaling[::2]/timeScaling[-2], timeScaling[1::2]/timeScaling[-1])
  ysfreq = sfreq[0,:]                                         # initialize output frame
  ysmag = smag[0,:]                                           # initialize output frame
  indexes = (L-1)*timeScalingEnv(np.arange(outL)/float(outL))

  L = hfreq[:,0].size                                           # number of analysis frames
  nH = hfreq[0,:].size                                          # number of harmonics
  yhfreq = hfreq[0,:]                                           # initialize output frame
  yhmag = hmag[0,:]                                            # initialize output frame
  ystocEnv = stocEnv[0,:]                                      # initialize output frame
  outL = int(max(outTime)/float(max(inTime))*L)                # number of synthesis frames
  outIndexes = (outL-1)*outTime/max(outTime)                   # output indexes
  inIndexes = (L-1)*inTime/max(inTime)                         # input indexes
  interpf = interp1d(outIndexes, inIndexes)                    # generate interpolation function
  il = np.arange(outL)
  indexes = interpf(il)
  for l in indexes[1:]:
    ystocEnv = np.vstack((ystocEnv, stocEnv[round(l),:]))
  return yhfreq, yhmag, ystocEnv

if __name__ == '__main__':
  (fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase.wav'))
  w = np.blackman(601)
  N = 1024
  t = -100
  nH = 100
  minf0 = 350
  maxf0 = 700
  f0et = 5
  maxnpeaksTwm = 5
  minSineDur = .1
  harmDevSlope = 0.01
  Ns = 512
  H = Ns/4
  stocf = .2
  hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
  inTime = np.array([0, 1])
  outTime = np.array([0, 2])            
  yhfreq, yhmag, ystocEnv = hpsTimeScale(hfreq, hmag, mYst, inTime, outTime)
  y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)
  UF.play(y, fs)


