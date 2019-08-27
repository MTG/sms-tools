# transformations applied to the harmonics of a sound

import numpy as np
from scipy.signal import resample
from scipy.interpolate import interp1d

def harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs):
	"""
	Frequency scaling of the harmonics of a sound
	hfreq, hmag: frequencies and magnitudes of input harmonics
	freqScaling: scaling factors, in time-value pairs (value of 1 no scaling)
	freqStretching: stretching factors, in time-value pairs (value of 1 no stretching)
	timbrePreservation: 0  no timbre preservation, 1 timbre preservation 
	fs: sampling rate of input sound
	returns yhfreq, yhmag: frequencies and magnitudes of output harmonics
	"""
	if (freqScaling.size % 2 != 0):                        # raise exception if array not even length
		raise ValueError("Frequency scaling array does not have an even size")
	
	if (freqStretching.size % 2 != 0):                     # raise exception if array not even length
		raise ValueError("Frequency stretching array does not have an even size")
		
	L = hfreq.shape[0]                                                   # number of frames
	nHarms = hfreq.shape[1]                                              # number of harmonics
	# create interpolation object with the scaling values 
	freqScalingEnv = np.interp(np.arange(L), L*freqScaling[::2]/freqScaling[-2], freqScaling[1::2]) 
	# create interpolation object with the stretching values
	freqStretchingEnv = np.interp(np.arange(L), L*freqStretching[::2]/freqStretching[-2], freqStretching[1::2]) 
	yhfreq = np.zeros_like(hfreq)                                        # create empty output matrix
	yhmag = np.zeros_like(hmag)                                          # create empty output matrix
	for l in range(L):                                                   # go through all frames
		ind_valid = np.where(hfreq[l,:]!=0)[0]                             # check if there are frequency values
		if ind_valid.size == 0:                                            # if no values go to next frame
			continue
		if (timbrePreservation == 1) & (ind_valid.size > 1):               # create spectral envelope
			# values of harmonic locations to be considered for interpolation
			x_vals = np.append(np.append(0, hfreq[l,ind_valid]),fs/2)     
			# values of harmonic magnitudes to be considered for interpolation 
			y_vals = np.append(np.append(hmag[l,0], hmag[l,ind_valid]),hmag[l,-1])     
			specEnvelope = interp1d(x_vals, y_vals, kind = 'linear',bounds_error=False, fill_value=-100)
		yhfreq[l,ind_valid] = hfreq[l,ind_valid] * freqScalingEnv[l]       # scale frequencies
		yhfreq[l,ind_valid] = yhfreq[l,ind_valid] * (freqStretchingEnv[l]**ind_valid) # stretch frequencies
		if (timbrePreservation == 1) & (ind_valid.size > 1):               # if timbre preservation
			yhmag[l,ind_valid] = specEnvelope(yhfreq[l,ind_valid])           # change amplitudes to maintain timbre
		else:
			yhmag[l,ind_valid] = hmag[l,ind_valid]                           # use same amplitudes as input
	return yhfreq, yhmag
	
