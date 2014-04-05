import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
from scipy.interpolate import interp1d

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import hpsModel as HPS
import utilFunctions as UF
import harmonicTransformations as HT

def hpsTimeScale(hfreq, hmag, stocEnv, timeScaling):
  # time scaling of the harmonic plus stochastic representation
  # hfreq, hmag: harmonic frequencies and magnitudes, stocEnv: residual envelope
  # timeScaling: scaling factors, in time-value pairs
  # returns yhfreq, yhmag: harmonic frequencies and amplitudes, ystocEnv: residual envelope
  L = hfreq[:,0].size                                         # number of analysis frames
  outL = int(L*timeScaling[-1]/timeScaling[-2])               # number of synthesis frames
  timeScalingEnv = interp1d(timeScaling[::2]/timeScaling[-2], timeScaling[1::2]/timeScaling[-1])
  indexes = (L-1)*timeScalingEnv(np.arange(outL)/float(outL))
  yhfreq = hfreq[0,:]                                          # first output frame is same than input
  yhmag = hmag[0,:]                                            # first output frame is same than input
  ystocEnv = stocEnv[0,:]                                      # first output frame is same than input
  for l in indexes[1:]:
    yhfreq = np.vstack((yhfreq, hfreq[round(l),:]))
    yhmag = np.vstack((yhmag, hmag[round(l),:])) 
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
  minSineDur = .1
  harmDevSlope = 0.01
  Ns = 512
  H = Ns/4
  stocf = .2
  hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
  freqScaling = np.array([0, 3, 1, .5])
  freqStretching = np.array([])
  timbrePreservation = 0
  hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

  timeScaling = np.array([0, 0, 1, 2])          
  yhfreq, yhmag, ystocEnv = hpsTimeScale(hfreqt, hmagt, mYst, timeScaling)
  y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)
  UF.play(y, fs)


