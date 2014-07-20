# transformations applyed to the harmonics of a sound

import numpy as np
from scipy.signal import resample
from scipy.interpolate import interp1d
import math, sys, os, functools, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import harmonicModel as HM
import sineModel as SM
import sineTransformations as ST

def harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs):
	# frequency scaling of harmonics
	# hfreq, hmag: frequencies and magnitudes of input harmonics
	# freqScaling: scaling factors, in time-value pairs
	# freqStretching: stretching factors, in time-value pairs
	# timbrePreservation: wether or not to use timbre preservation
	# fs: sampling rate
	# returns yhfreq, yhmag: frequencies and magnitudes of output harmonics
	L = hfreq[:,0].size                                     # number of frames
	nHarms = hfreq[0,:].size                                # number of harmonics
	freqScaling = np.interp(np.arange(L), L*freqScaling[::2]/freqScaling[-2], freqScaling[1::2]) 
	if (freqStretching.size>0):
		freqStretching = np.interp(np.arange(L), L*freqStretching[::2]/freqStretching[-2], freqStretching[1::2]) 
	yhfreq = hfreq
	yhmag = hmag
	for l in range(L):                                      # go through all frames
		ind_valid = np.where(yhfreq[l,:]!=0)[0]
		if ind_valid.size == 0:
			continue
		if (timbrePreservation == 1) & (ind_valid.size > 1):
			x_vals = np.append(np.append(0, yhfreq[l,ind_valid]),fs/2)      # values of peak locations to be considered for interpolation
			y_vals = np.append(np.append(yhmag[l,0], yhmag[l,ind_valid]),yhmag[l,-1])     # values of peak magnitudes to be considered for interpolation
			specEnvelope = interp1d(x_vals, y_vals, kind = 'linear',bounds_error=False, fill_value=-100)
		yhfreq[l,ind_valid] = yhfreq[l,ind_valid] * freqScaling[l]
		if (timbrePreservation == 1) & (ind_valid.size > 1):
			yhmag[l,ind_valid] = specEnvelope(yhfreq[l,ind_valid])
		if freqStretching.size > 0:
			yhfreq[l,ind_valid] = yhfreq[l,ind_valid] * (freqStretching[l]**ind_valid)
	return yhfreq, yhmag

########## not finished #####################
def harmonicModulation(hfreq, hmag, frameRate, modRate, vibDepth, tremDepth):
	# vibrato and tremolo modulation of harmonics
	# hfreq, hmag: frequencies and magnitudes of input harmonics
	# modRate, vibDepth, tremDepth: modulation rate, vibrato depth and tremolo depth
	# returns yhfreq, yhmag: frequencies and magnitudes of output harmonics
	L = hfreq[:,0].size                                     # number of frames
	nHarms = hfreq[0,:].size                                # number of harmonics
	modRate = np.interp(np.arange(L), L*modRate[::2]/modRate[-2], modRate[1::2])
	vibDepth = np.interp(np.arange(L), L*vibDepth[::2]/vibDepth[-2], vibDepth[1::2])
	tremDepth = np.interp(np.arange(L), L*tremDepth[::2]/tremDepth[-2], tremDepth[1::2])
	modf = np.sin(2.0*np.pi*modRate/frameRate)              # modulation factor for both vibrato and tremolo (which has to be scaled later)
	sfscale = fscale*(2.0**(vd/1200.0*modf))                # affective scale factor together with vibrato affect
 
	yhfreq = hfreq
	yhmag = hmag
	for l in range(L):                                      # go through all frames
		yhfreq[l,ind_valid] = yhfreq[l,ind_valid] * freqScaling[l]
	return yhfreq, yhmag
