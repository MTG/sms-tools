# functions that implement analysis and synthesis of sounds using the Short-Time Fourier Transform
# (for example usage check the examples directory)

import numpy as np
from scipy.signal import hamming
import math
import dftModel as DFT
import utilFunctions as UF

def stft(x, fs, w, N, H):
# analysis/synthesis of a sound using the short-time fourier transform
# x: input sound, w: analysis window, N: FFT size, H: hop size
# returns y: output sound
	M = w.size                                     # size of analysis window
	hM1 = int(math.floor((M+1)/2))                 # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                     # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                 # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM1))                 # add zeros at the end to analyze last sample
	pin = hM1                                      # initialize sound pointer in middle of analysis window       
	pend = x.size-hM1                              # last sample to start a frame
	w = w / sum(w)                                 # normalize analysis window
	y = np.zeros(x.size)                           # initialize output array
	while pin<=pend:                               # while sound pointer is smaller than last sample      
	#-----analysis-----  
		x1 = x[pin-hM1:pin+hM2]                      # select one frame of input sound
		mX, pX = DFT.dftAnal(x1, w, N)               # compute dft
	#-----synthesis-----
		y1 = DFT.dftSynth(mX, pX, M)                 # compute idft
		y[pin-hM1:pin+hM2] += H*y1                   # overlap-add to generate output sound
		pin += H                                     # advance sound pointer
	y = np.delete(y, range(hM2))                   # delete half of first window which was added in stftAnal
	y = np.delete(y, range(y.size-hM1, y.size))    # delete half of the last window which as added in stftAnal
	return y

def stftAnal(x, fs, w, N, H) :
	# analysis of a sound using the short-time fourier transform
	# x: input array sound, w: analysis window, N: FFT size, H: hop size
	# returns xmX, xpX: magnitude and phase spectra
	M = w.size                                      # size of analysis window
	hM1 = int(math.floor((M+1)/2))                  # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                      # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                  # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                  # add zeros at the end to analyze last sample
	pin = hM1                                       # initialize sound pointer in middle of analysis window       
	pend = x.size-hM1                               # last sample to start a frame
	w = w / sum(w)                                  # normalize analysis window
	y = np.zeros(x.size)                            # initialize output array
	while pin<=pend:                                # while sound pointer is smaller than last sample      
		x1 = x[pin-hM1:pin+hM2]                     # select one frame of input sound
		mX, pX = DFT.dftAnal(x1, w, N)              # compute dft
		if pin == hM1: 
			xmX = np.array([mX])
			xpX = np.array([pX])
		else:
			xmX = np.vstack((xmX,np.array([mX])))
			xpX = np.vstack((xpX,np.array([pX])))
		pin += H                                   # advance sound pointer
	return xmX, xpX

def stftSynth(mY, pY, M, H) :
# synthesis of a sound using the short-time fourier transform
# mY: magnitude spectra, pY: phase spectra, M: window size, H: hop-size
# returns y: output sound
	hM1 = int(math.floor((M+1)/2))                   # half analysis window size by rounding
	hM2 = int(math.floor(M/2))                       # half analysis window size by floor
	nFrames = mY[:,0].size                           # number of frames
	y = np.zeros(nFrames*H + hM1 + hM2)              # initialize output array
	pin = hM1                  
	for i in range(nFrames):                         # iterate over all frames      
		y1 = DFT.dftSynth(mY[i,:], pY[i,:], M)       # compute idft
		y[pin-hM1:pin+hM2] += H*y1                   # overlap-add to generate output sound
		pin += H                                     # advance sound pointer
	y = np.delete(y, range(hM2))                     # delete half of first window which was added in stftAnal
	y = np.delete(y, range(y.size-hM1, y.size))      # add zeros at the end to analyze last sample
	return y

 
