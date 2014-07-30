# functions that implement transformations using the hpsModel

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import hpsModel as HPS
import utilFunctions as UF

def hpsTimeScale(hfreq, hmag, stocEnv, timeScaling):
	# time scaling of the harmonic plus stochastic representation
	# hfreq, hmag: harmonic frequencies and magnitudes, stocEnv: residual envelope
	# timeScaling: scaling factors, in time-value pairs
	# returns yhfreq, yhmag, ystocEnv: hps output representation
	L = hfreq[:,0].size                                    # number of input frames
	maxInTime = max(timeScaling[::2])                      # maximum value used as input times
	maxOutTime = max(timeScaling[1::2])                    # maximum value used in output times
	outL = int(L*maxOutTime/maxInTime)                     # number of output frames
	inFrames = (L-1)*timeScaling[::2]/maxInTime                # input time values in frames
	outFrames = outL*timeScaling[1::2]/maxOutTime          # output time values in frames
	timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)    # interpolation function
	indexes = timeScalingEnv(np.arange(outL))              # generate frame indexes for the output
	yhfreq = hfreq[round(indexes[0]),:]                   # first output frame
	yhmag = hmag[round(indexes[0]),:]                     # first output frame
	ystocEnv = stocEnv[round(indexes[0]),:]                # first output frame
	for l in indexes[1:]:
		yhfreq = np.vstack((yhfreq, hfreq[round(l),:]))
		yhmag = np.vstack((yhmag, hmag[round(l),:])) 
		ystocEnv = np.vstack((ystocEnv, stocEnv[round(l),:]))
	return yhfreq, yhmag, ystocEnv
	
	
def hpsMorph(hfreq1, hmag1, stocEnv1, hfreq2, hmag2, stocEnv2, hfreqIntp, hmagIntp, stocIntp):
  # morph between two sounds using the harmonic plus stochastic model
  # hfreq1, hmag1, stocEnv1: hps representation of sound 1
  # hfreq2, hmag2, stocEnv2: hps representation of sound 2
	# hfreqIntp: interpolation factor between the harmonic frequencies of the two sounds, 0 is sound 1 and 1 is sound 2 (time,value pairs)
	# hmagIntp: interpolation factor between the harmonic magnitudes of the two sounds, 0 is sound 1 and 1 is sound 2  (time,value pairs)
	# stocIntp: interpolation factor between the stochastic representation of the two sounds, 0 is sound 1 and 1 is sound 2  (time,value pairs)
  # returns yhfreq, yhmag, ystocEnv: hps output representation
	L1 = hfreq1[:,0].size                                  # number of frames of sound 1
	L2 =  hfreq2[:,0].size                                 # number of frames of sound 2
	hfreqIntp[::2] = (L1-1)*hfreqIntp[::2]/hfreqIntp[-2]   # normalize input values
	hmagIntp[::2] = (L1-1)*hmagIntp[::2]/hmagIntp[-2]      # normalize input values
	stocIntp[::2] = (L1-1)*stocIntp[::2]/stocIntp[-2]      # normalize input values
	hfreqIntpEnv = interp1d(hfreqIntp[0::2], hfreqIntp[1::2], fill_value=0)    # interpolation function
	hfreqIndexes = hfreqIntpEnv(np.arange(L1))             # generate frame indexes for the output
	hmagIntpEnv = interp1d(hmagIntp[0::2], hmagIntp[1::2], fill_value=0)    # interpolation function
	hmagIndexes = hmagIntpEnv(np.arange(L1))               # generate frame indexes for the output
	stocIntpEnv = interp1d(stocIntp[0::2], stocIntp[1::2], fill_value=0)    # interpolation function
	stocIndexes = stocIntpEnv(np.arange(L1))               # generate frame indexes for the output
	yhfreq = np.zeros_like(hfreq1)                         # create empty output matrix
	yhmag = np.zeros_like(hmag1)                           # create empty output matrix
	ystocEnv = np.zeros_like(stocEnv1)                     # create empty output matrix
	for l in range(L1):                                    # generate morphed frames
		if (hfreqIndexes[l] == 0):                           # if factor is 0 use values of sound 1
			yhfreq[l,:] = hfreq1[l,:]
			yhmag[l,:] = hmag1[l,:]
		elif (hfreqIndexes[l] == 1):                         # if factor is 1 use values of sound 2
			yhfreq[l,:] = hfreq2[round(L2*l/float(L1)),:]
			yhmag[l,:] = hmag2[round(L2*l/float(L1)),:]
		else:                                                # otherwise perform the appropiate interpolation
			harmonics = np.intersect1d(np.array(np.nonzero(hfreq1[l,:]), dtype=np.int)[0], np.array(np.nonzero(hfreq2[round(L2*l/float(L1)),:]), dtype=np.int)[0])
			yhfreq[l,harmonics] =  (1-hfreqIndexes[l])* hfreq1[l,harmonics] + hfreqIndexes[l] * hfreq2[round(L2*l/float(L1)),harmonics]
			yhmag[l,harmonics] =  (1-hmagIndexes[l])* hmag1[l,harmonics] + hmagIndexes[l] * hmag2[round(L2*l/float(L1)),harmonics]
		ystocEnv[l,:] =  (1-stocIndexes[l])* stocEnv1[l,:] + stocIndexes[l] * stocEnv2[round(L2*l/float(L1)),:]
	return yhfreq, yhmag, ystocEnv
  

