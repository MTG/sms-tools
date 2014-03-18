import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal
import waveIO as WIO
import peakProcessing as PP
import harmonicDetection as HD
import errorHandler as EH
import harmonicModelAnal as HA
import stochasticResidual as SR

try:
	import genSpecSines_C as GS
except ImportError:
	import genSpecSines as GS
	EH.printWarning(1)
	
def hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur, Ns, stocf):
	# Analysis of a sound using the harmonic plus stochastic model
	# x: input sound, fs: sampling rate, w: analysis window, 
	# N: FFT size, t: threshold in negative dB, 
	# nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
	# maxf0: maximim f0 frequency in Hz, 
	# f0et: error threshold in the f0 detection (ex: 5),
	# harmDevSlope: slope of harmonic deviation
	# maxnpeaksTwm: maximum number of peaks used for F0 detection
	# minSineDur: minimum length of harmonics
	# returns: hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; mYst: stochastic residual
	hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, 
																						 maxnpeaksTwm, minSineDur)
	mYst = SR.stochasticResidual(x, Ns, H, hfreq, hmag, hphase, fs, stocf)
	return hfreq, hmag, hphase, mYst

if __name__ == '__main__':
	(fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase.wav'))
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
	hfreq, hmag, hphase, mYst = hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, 
																						maxnpeaksTwm, minSineDur, Ns, stocf)

	maxplotfreq = 20000.0
	numFrames = int(mYst[:,0].size)
	sizeEnv = int(mYst[0,:].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:sizeEnv*maxplotfreq/(.5*fs)+1]))
	plt.autoscale(tight=True)

	harms = hfreq*np.less(hfreq,maxplotfreq)
	harms[harms==0] = np.nan
	numFrames = int(harms[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs) 
	plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
	plt.autoscale(tight=True)
	plt.title('harmonic + stochastic components')
	plt.show()



