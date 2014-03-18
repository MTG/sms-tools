import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import errorHandler as EH

try:
	import genSpecSines_C as GS
except ImportError:
	import genSpecSines as GS
	EH.printWarning(1)
	
def stochasticResidual(x, N, H, sfreq, smag, sphase, fs, stocf):
	# subtract sinusoids from a sound
	# x: input sound, N: fft-size, H: hop-size
	# sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
	# returns mYst: stochastic approximation of residual 
	hN = N/2  
	x = np.append(np.zeros(hN),x)                    # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hN))                    # add zeros at the end to analyze last sample
	bh = blackmanharris(N)                           # synthesis window
	w = bh/ sum(bh)                                  # normalize synthesis window
	L = sfreq[:,0].size                              # number of frames   
	pin = 0
	for l in range(L):
		xw = x[pin:pin+N]*w                            # window the input sound                               
		X = fft(fftshift(xw))                          # compute FFT 
		Yh = GS.genSpecSines(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N)   # generate spec sines          
		Xr = X-Yh                                      # subtract sines from original spectrum
		mXr = 20*np.log10(abs(Xr[:hN]))                # magnitude spectrum of residual
		mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)  # decimate the mag spectrum                        
		if l == 0: 
			mYst = np.array([mXrenv])
		else:
			mYst = np.vstack((mYst, np.array([mXrenv])))
		pin += H   									                   # advance sound pointer
	return mYst