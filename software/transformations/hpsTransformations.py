# functions that implement time scale transformations using the hpsModel

import numpy as np
from scipy.interpolate import interp1d

def hpsTimeScale(hfreq, hmag, stocEnv, timeScaling):
	# time scaling of the harmonic plus stochastic representation
	# hfreq, hmag: harmonic frequencies and magnitudes, stocEnv: residual envelope
	# timeScaling: scaling factors, in time-value pairs
	# returns yhfreq, yhmag: harmonic frequencies and amplitudes, ystocEnv: residual envelope
	L = hfreq[:,0].size                                    # number of input frames
	maxInTime = max(timeScaling[::2])                      # maximum value used as input times
	maxOutTime = max(timeScaling[1::2])                    # maximum value used in output times
	outL = int(L*maxOutTime/maxInTime)                     # number of output frames
	inFrames = (L-1)*timeScaling[::2]/maxInTime                # input time values in frames
	outFrames = outL*timeScaling[1::2]/maxOutTime          # output time values in frames
	timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)    # interpolation function
	indexes = timeScalingEnv(np.arange(outL))              # generate frame indexes for the output
	yhfreq = hfreq[round(indexes[0]),:]                    # first output frame
	yhmag = hmag[round(indexes[0]),:]                      # first output frame
	ystocEnv = stocEnv[round(indexes[0]),:]                # first output frame
	for l in indexes[1:]:
		yhfreq = np.vstack((yhfreq, hfreq[round(l),:]))
		yhmag = np.vstack((yhmag, hmag[round(l),:])) 
		ystocEnv = np.vstack((ystocEnv, stocEnv[round(l),:]))
	return yhfreq, yhmag, ystocEnv
