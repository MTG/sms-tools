import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))

import stftAnal
import waveIO as WIO
import harmonicModelAnal as HA
import harmonicModelSynth as HS
import sineSubtraction as SS
	

if __name__ == '__main__':
	(fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase.wav'))
	w = np.blackman(551)
	N = 1024
	t = -100
	nH = 100
	minf0 = 350
	maxf0 = 700
	f0et = 5
	maxnpeaksTwm = 5
	minSineDur = .1
	harmDevSlope = 0.01
	Ns = 512
	H = Ns/4
	hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
	xr = SS.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
	yh = HS.harmonicModelSynth(hfreq, hmag, hphase, Ns, H, fs)

	WIO.play(yh, fs)
	WIO.play(xr, fs)
