# functions that implement transformations using the hpsModel

import numpy as np
from scipy.interpolate import interp1d
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import stochasticModel as STC
import utilFunctions as UF

def stochasticTimeScale(stocEnv, timeScaling):
  # time scaling of the stochastic representation of a sound
  # stocEnv: stochastic envelope
  # timeScaling: scaling factors, in time-value pairs
  # returns ystocEnv: stochastic envelope
  L = stocEnv[:,0].size                                       # number of input frames
  outL = int(L*timeScaling[-1]/timeScaling[-2])               # number of synthesis frames
  timeScalingEnv = interp1d(timeScaling[::2]/timeScaling[-2], timeScaling[1::2]/timeScaling[-1])
  indexes = (L-1)*timeScalingEnv(np.arange(outL)/float(outL))
  ystocEnv = stocEnv[0,:]                                     # first output frame is same than input
  for l in indexes[1:]:
    ystocEnv = np.vstack((ystocEnv, stocEnv[round(l),:]))
  return ystocEnv
