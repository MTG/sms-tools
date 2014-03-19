import numpy as np
import matplotlib.pyplot as plt
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import sineModelAnal as SA 
import sineModelSynth as SS
import waveIO as WIO


def sineModelFreqScale(sfreq, freqScaling):
	# frequency scaling of sinusoids
	# sfreq: frequencies of input sinusoids
	# freqScaling: scaling factors, in time-value pairs
	# returns sfreq: frequencies output sinusoids
	L = sfreq[:,0].size            # number of frames
	freqScaling = np.interp(np.arange(L), L*freqScaling[::2]/freqScaling[-2], freqScaling[1::2]) 
	ysfreq = np.empty_like(sfreq)  # create empty output matrix
	for l in range(L):             # go through all frames
		ysfreq[l,:] = sfreq[l,:] * freqScaling[l]
	return ysfreq

if __name__ == '__main__':
	(fs, x) = WIO.wavread('../../sounds/orchestra.wav')
	w = np.hamming(2000)
	N = 2048
	H = 500
	t = -90
	minSineDur = .01
	maxnSines = 150
	freqDevOffset = 20
	freqDevSlope = 0.02
	Ns = 512
	H = Ns/4
	tfreq, tmag, tphase = SA.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
	freqScaling = np.array([0, .8, 1, 1.2])           
	ytfreq = sineModelFreqScale(tfreq, freqScaling)
	y = SS.sineModelSynth(ytfreq, tmag, np.array([]), Ns, H, fs)
	WIO.play(y, fs)


