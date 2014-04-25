import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math, sys, os, functools, time
from scipy.interpolate import interp1d
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import sineModel as SM 
import utilFunctions as UF

def sineTimeScaling(sfreq, smag, timeScaling):
  # time scaling of sinusoidal tracks
  # sfreq, smag: frequencies and magnitudes of input sinusoidal tracks
  # timeScaling: scaling factors, in time-value pairs
  # returns ysfreq, ysmag: frequencies and magnitudes of output sinusoidal tracks
  L = sfreq[:,0].size                                    # number of input frames
  maxInTime = max(timeScaling[::2])                      # maximum value used as input times
  maxOutTime = max(timeScaling[1::2])                    # maximum value used in output times
  outL = int(L*maxOutTime/maxInTime)                     # number of output frames
  inFrames = L*timeScaling[::2]/maxInTime                # input time values in frames
  outFrames = outL*timeScaling[1::2]/maxOutTime          # output time values in frames
  timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)    # interpolation function
  indexes = timeScalingEnv(np.arange(outL))              # generate frame indexes for the output
  ysfreq = sfreq[round(indexes[0]),:]                    # first output frame
  ysmag = smag[round(indexes[0]),:]                      # first output frame
  for l in indexes[1:]:                                  # generate frames for output sine tracks
    ysfreq = np.vstack((ysfreq, sfreq[round(l),:]))
    ysmag = np.vstack((ysmag, smag[round(l),:])) 
  return ysfreq, ysmag

def sineFreqScaling(sfreq, freqScaling):
  # frequency scaling of sinusoidal tracks
  # sfreq: frequencies of input sinusoidal tracks
  # freqScaling: scaling factors, in time-value pairs
  # returns sfreq: frequencies of output sinusoidal tracks
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
  timeScale = np.array([.01, .0, .02, .02, .335, .4, .345, .41, .671, .8, .681, .81, .858, 1.2, .868, 1.21, 1.185, 1.6, 1.195, 1.61, 1.497, 2.0, 1.507, 2.01, 1.686, 2.4, 1.696, 2.41, 1.978, 2.8])          
  ytfreq, ytmag = sineTimeScaling(ytfreq, tmag, timeScale)
  y = SM.sineModelSynth(ytfreq, ytmag, np.array([]), Ns, H, fs)
  UF.play(y, fs)


