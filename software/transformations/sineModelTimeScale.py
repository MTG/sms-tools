import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
from scipy.interpolate import interp1d

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import sineModelAnal as SA 
import sineModelSynth as SS
import waveIO as WIO


def sineModelTimeScale(sfreq, smag, inTime, outTime):
  # time scaling of sinusoids
  # sfreq, smag: frequencies and magnitudes of input sinusoids
  # inTime: array of input times, outTime: array of output times
  # ysfreq, ysmag: frequencies and magnitudes of output sinusoids
  l = 0                                                        # frame index
  L = sfreq[:,0].size                                          # number of analysis frames
  nT = sfreq[0,:].size                                         # number of tracks
  ysfreq = sfreq[0,:]                                          # initialize output frame
  ysmag = smag[0,:]                                            # initialize output frame
  outL = int(max(outTime)/float(max(inTime))*L)                # number of synthesis frames
  outIndexes = (outL-1)*outTime/max(outTime)                   # output indexes
  inIndexes = (L-1)*inTime/max(inTime)                         # input indexes
  interpf = interp1d(outIndexes, inIndexes)                    # generate interpolation function
  il = np.arange(outL)
  indexes = interpf(il)
  for l in indexes[1:]:
    ysfreq = np.vstack((ysfreq, sfreq[round(l),:]))
    ysmag = np.vstack((ysmag, smag[round(l),:])) 
  return ysfreq, ysmag, indexes

if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/mridangam.wav')
  w = np.hamming(801)
  N = 2048
  t = -90
  minSineDur = .005
  maxnSines = 150
  freqDevOffset = 20
  freqDevSlope = 0.02
  Ns = 512
  H = Ns/4
  tfreq, tmag, tphase = SA.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
  inTime = np.array([0, .091, .405, .747, .934, 1.259, 1.568, 1.761, 2.057])
  outTime = np.array([0, .091, .405+.4, .747+.4, .934+.4, 1.259+.8, 1.568+.8, 1.761+1.2, 2.057+1.2])            
  ytfreq, ytmag, indexes = sineModelTimeScale(tfreq, tmag, inTime, outTime)
  y = SS.sineModelSynth(ytfreq, ytmag, np.array([]), Ns, H, fs)
  WIO.play(y, fs)


