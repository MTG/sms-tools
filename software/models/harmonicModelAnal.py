import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import math
import sys, os, functools
import operator as op

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as WIO
import peakProcessing as PP
import harmonicDetection as HD
import errorHandler as EH
import stftAnal, dftAnal
import sineTracking as ST

try:
	import genSpecSines_C as GS
	import twm_C as TWM
except ImportError:
	import genSpecSines as GS
	import twm as TWM
	EH.printWarning(1)

def harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope=0.01, maxnpeaksTwm=10, minSineDur=.02):
	# Analysis of a sound using the sinusoidal harmonic model
	# x: input sound, fs: sampling rate, w: analysis window, 
	# N: FFT size (minimum 512), t: threshold in negative dB, 
	# nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
	# maxf0: maximim f0 frequency in Hz, 
	# f0et: error threshold in the f0 detection (ex: 5),
	# harmDevSlope: slope of harmonic deviation
	# maxnpeaksTwm: maximum number of peaks used for F0 detection
	# minSineDur: minimum length of harmonics
	# returns xhfreq, xhmag, xhphase: harmonic frequencies, magnitudes and phases
	hN = N/2                                                # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
	pin = hM1                                               # init sound pointer in middle of anal window          
	pend = x.size - hM1                                     # last sample to start a frame
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	w = w / sum(w)                                          # normalize analysis window
	hfreqp = []
	while pin<=pend:           
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft            
		ploc = PP.peakDetection(mX, hN, t)                    # detect peak locations   
		iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values
		ipfreq = fs * iploc/N
		f0 = TWM.f0DetectionTwm(ipfreq, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
		hfreq, hmag, hphase = HD.harmonicDetection(ipfreq, ipmag, ipphase, f0, nH, hfreqp, fs, harmDevSlope) # find harmonics
		hfreqp = hfreq
		if pin == hM1: 
			xhfreq = np.array([hfreq])
			xhmag = np.array([hmag])
			xhphase = np.array([hphase])
		else:
			xhfreq = np.vstack((xhfreq,np.array([hfreq])))
			xhmag = np.vstack((xhmag, np.array([hmag])))
			xhphase = np.vstack((xhphase, np.array([hphase])))
		pin += H                                              # advance sound pointer
	xhfreq = ST.cleaningSineTracks(xhfreq, round(fs*minSineDur/H))
	return xhfreq, xhmag, xhphase

if __name__ == '__main__':
	(fs, x) = WIO.wavread('../../sounds/vignesh.wav')
	w = np.blackman(1201)
	N = 2048
	t = -90
	nH = 100
	minf0 = 130
	maxf0 = 300
	f0et = 7
	maxnpeaksTwm = 4
	Ns = 512
	H = Ns/4
	minSineDur = .1
	harmDevSlope = 0.01

	mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
	hfreq, hmag, hphase = harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
	maxplotfreq = 20000.0
	numFrames = int(mX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
	plt.autoscale(tight=True)
	
	harms = hfreq*np.less(hfreq,maxplotfreq)
	harms[harms==0] = np.nan
	numFrames = int(hfreq[:,0].size)
	plt.plot(frmTime, harms, color='k')
	plt.autoscale(tight=True)
	plt.title('harmonics on spectrogram')
	plt.show()

