import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/transformations/'))

import hpsModel as HPS
import hpsTransformations as HPST
import harmonicTransformations as HT
import utilFunctions as UF

if __name__ == '__main__':
	(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/sax-phrase-short.wav'))
	w = np.blackman(601)
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
	stocf = .2
	hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
	freqScaling = np.array([0, 1.2, 1.18, 1.2, 1.89, 1.2, 2.01, 1.2, 2.679, .7, 3.146, .7])
	freqStretching = np.array([])
	timbrePreservation = 1
	hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

	timeScaling = np.array([0, 0, 0.073, 0.073, 0.476, 0.476-0.2, 0.512, .512-0.2, 0.691, 0.691+0.2, 1.14, 1.14, 1.21, 1.21, 1.87, 1.87-0.4, 2.138, 2.138-0.4, 2.657, 2.657+.8, 2.732, 2.732+.8, 2.91, 2.91+.7, 3.146, 3.146+.7])
	yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreqt, hmagt, mYst, timeScaling)

	y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)
	UF.play(y, fs)
	UF.wavwrite(y,fs,'sax-phrase-short-synthesis.wav')
		# UF.wavwrite(yh,fs,'sax-phrase-harmonic-component.wav')
		# UF.wavwrite(yr,fs,'sax-phrase-residual-component.wav')
		# UF.wavwrite(yst,fs,'sax-phrase-stochastic-component.wav')
