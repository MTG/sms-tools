# functions that implement transformations using the hpsModel

import numpy as np
from scipy.interpolate import interp1d


def stochasticTimeScale(stocEnv, timeScaling):
	"""
	Time scaling of the stochastic representation of a sound
	stocEnv: stochastic envelope
	timeScaling: scaling factors, in time-value pairs
	returns ystocEnv: stochastic envelope
	"""
	if (timeScaling.size % 2 != 0):                             # raise exception if array not even length
		raise ValueError("Time scaling array does not have an even size")
		
	L = stocEnv[:,0].size                                       # number of input frames
	outL = int(L*timeScaling[-1]/timeScaling[-2])               # number of synthesis frames
	# create interpolation object with the time scaling values
	timeScalingEnv = interp1d(timeScaling[::2]/timeScaling[-2], timeScaling[1::2]/timeScaling[-1])
	indexes = (L-1)*timeScalingEnv(np.arange(outL)/float(outL)) # generate output time indexes
	ystocEnv = stocEnv[0,:]                                     # first output frame is same than input
	for l in indexes[1:]:                                       # step through the output frames
		ystocEnv = np.vstack((ystocEnv, stocEnv[int(round(l)),:]))     # get the closest input frame
	return ystocEnv
