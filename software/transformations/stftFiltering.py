import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import dftModel as DFT 
import utilFunctions as UF
import math

def stftFiltering(x, fs, w, N, H, filter):
# Analysis/synthesis of a sound using the short-time fourier transform
# x: input sound, w: analysis window, N: FFT size, H: hop size
# filter: magnitude response of filter with frequency-magnitude pairs (in dB)
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
	filt = np.interp(np.arange(N/2), (N/2)*filter[::2]/filter[-2], filter[1::2])  # generate filter shape
	while pin<=pend:                               # while sound pointer is smaller than last sample      
	#-----analysis-----  
		x1 = x[pin-hM1:pin+hM2]                      # select one frame of input sound
		mX, pX = DFT.dftAnal(x1, w, N)               # compute dft
	# filter
		mY = mX + filt                               # filter input magnitude spectrum
	#-----synthesis-----
		y1 = DFT.dftSynth(mY, pX, M)                # compute idft
		y[pin-hM1:pin+hM2] += H*y1                  # overlap-add to generate output sound
		pin += H                                    # advance sound pointer
	y = np.delete(y, range(hM2))                   # delete half of first window which was added in stftAnal
	y = np.delete(y, range(y.size-hM1, y.size))    # add zeros at the end to analyze last sample
	return y


# example call of stft function
if __name__ == '__main__':
	(fs, x) = UF.wavread('../../sounds/orchestra.wav')
	w = np.hamming(500)
	N = 1024
	H = 100
	filter = np.array([0,0,500, 00, 600,-20,800,-40,22050,-100])
	y = stftFiltering(x, fs, w, N, H, filter)
	UF.play(y, fs)
