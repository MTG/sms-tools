# functions that implement transformations using the sineModel

import numpy as np
from scipy.interpolate import interp1d

def sineTimeScaling(sfreq, smag, timeScaling):
  # time scaling of sinusoidal tracks
  # sfreq, smag: frequencies and magnitudes of input sinusoidal tracks
  # timeScaling: scaling factors, in time-value pairs
  # returns ysfreq, ysmag: frequencies and magnitudes of output sinusoidal tracks
  L = sfreq[:,0].size                                    # number of input frames
  maxInTime = max(timeScaling[::2])                      # maximum value used as input times
  maxOutTime = max(timeScaling[1::2])                    # maximum value used in output times
  outL = int(L*maxOutTime/maxInTime)                     # number of output frames
  inFrames = (L-1)*timeScaling[::2]/maxInTime            # input time values in frames
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
