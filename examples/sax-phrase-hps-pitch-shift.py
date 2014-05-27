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
import utilFunctions as UF

if __name__ == '__main__':
	(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/sax-phrase.wav'))
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
	inTime = np.array([0, 0.165, 0.595, 0.850, 1.15, 2.15, 2.81, 3.285, 4.585, 4.845, 5.1, 6.15, 6.825, 7.285, 8.185, 8.830, 9.379])
	outTime = np.array([0, 0.165, 0.595, 0.850, .9+1.15, 2.15, 2.81, 3.285, 4.585, 4.845, .9+5.1, 6.15, 6.825, 7.285, 8.185, 8.830, 9.379])            
	yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreq, hmag, mYst, inTime, outTime)
	y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)
	UF.play(y, fs)
		# wp.wavwrite(y,fs,'sax-phrase-total-synthesis.wav')
		# wp.wavwrite(yh,fs,'sax-phrase-harmonic-component.wav')
		# wp.wavwrite(yr,fs,'sax-phrase-residual-component.wav')
		# wp.wavwrite(yst,fs,'sax-phrase-stochastic-component.wav')
