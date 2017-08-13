# functions that implement transformations using the sineModel

import numpy as np
from scipy.interpolate import interp1d

def sineTimeScaling(sfreq, smag, timeScaling):
	"""
	Time scaling of sinusoidal tracks
	sfreq, smag: frequencies and magnitudes of input sinusoidal tracks
	timeScaling: scaling factors, in time-value pairs
	returns ysfreq, ysmag: frequencies and magnitudes of output sinusoidal tracks
	"""
	if (timeScaling.size % 2 != 0):                        # raise exception if array not even length
		raise ValueError("Time scaling array does not have an even size")
		
	L = sfreq.shape[0]                                     # number of input frames
	maxInTime = max(timeScaling[::2])                      # maximum value used as input times
	maxOutTime = max(timeScaling[1::2])                    # maximum value used in output times
	outL = int(L*maxOutTime/maxInTime)                     # number of output frames
	inFrames = (L-1)*timeScaling[::2]/maxInTime            # input time values in frames
	outFrames = outL*timeScaling[1::2]/maxOutTime          # output time values in frames
	timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)    # interpolation function
	indexes = timeScalingEnv(np.arange(outL))              # generate frame indexes for the output
	ysfreq = sfreq[int(round(indexes[0])),:]                    # first output frame
	ysmag = smag[int(round(indexes[0])),:]                      # first output frame
	for l in indexes[1:]:                                  # generate frames for output sine tracks
		ysfreq = np.vstack((ysfreq, sfreq[int(round(l)),:]))    # get closest frame to scaling value
		ysmag = np.vstack((ysmag, smag[int(round(l)),:]))       # get closest frame to scaling value
	return ysfreq, ysmag

def sineFreqScaling(sfreq, freqScaling):
	"""
	Frequency scaling of sinusoidal tracks
	sfreq: frequencies of input sinusoidal tracks
	freqScaling: scaling factors, in time-value pairs (value of 1 is no scaling)
	returns ysfreq: frequencies of output sinusoidal tracks
	"""
	if (freqScaling.size % 2 != 0):                        # raise exception if array not even length
		raise ValueError("Frequency scaling array does not have an even size")
		
	L = sfreq.shape[0]                                     # number of input frames
	# create interpolation object from the scaling values
	freqScalingEnv = np.interp(np.arange(L), L*freqScaling[::2]/freqScaling[-2], freqScaling[1::2]) 
	ysfreq = np.zeros_like(sfreq)                          # create empty output matrix
	for l in range(L):                                     # go through all frames
		ind_valid = np.where(sfreq[l,:]!=0)[0]               # check if there are frequency values
		if ind_valid.size == 0:                              # if no values go to next frame
			continue
		ysfreq[l,ind_valid] = sfreq[l,ind_valid] * freqScalingEnv[l] # scale of frequencies 
	return ysfreq
