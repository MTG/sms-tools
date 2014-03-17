import numpy as np

def harmonicDetection(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):
	# detection of the harmonics from a set of spectral peaks, finds the peaks that are closer
	# to the ideal harmonic series built on top of a fundamental frequency
	# pfreq, pmag, pphase: peak frequencies, magnitudes and phases
	# f0: fundamental frequency, nH: number of harmonics,
	# hfreqp: harmonic frequencies of previous frame,
	# fs: sampling rate, harmDevSlope: slope of change of the deviation allowed to perfect harmonic
	# returns hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases
	if (f0<=0):
		return np.zeros(nH), np.zeros(nH), np.zeros(nH)
	hfreq = np.zeros(nH)                                 # initialize harmonic frequencies
	hmag = np.zeros(nH)-100                              # initialize harmonic magnitudes
	hphase = np.zeros(nH)                                # initialize harmonic phases
	hf = f0*np.arange(1, nH+1)                           # initialize harmonic frequencies
	hi = 0                                               # initialize harmonic index
	if hfreqp == []:
		hfreqp = hf
	while (f0>0) and (hi<nH) and (hf[hi]<fs/2):          # find harmonic peaks
		pei = np.argmin(abs(pfreq - hf[hi]))               # closest peak
		dev1 = abs(pfreq[pei] - hf[hi])                    # deviation from perfect harmonic
		dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi]>0 else fs) # deviation from previous frame
		threshold = f0/3 + harmDevSlope * pfreq[pei]
		if ((dev1<threshold) or (dev2<threshold)):         # accept peak if deviation is small
			hfreq[hi] = pfreq[pei]                           # harmonic frequencies
			hmag[hi] = pmag[pei]                             # harmonic magnitudes
			hphase[hi] = pphase[pei]                         # harmonic phases
		hi += 1                                            # increase harmonic index
	return hfreq, hmag, hphase