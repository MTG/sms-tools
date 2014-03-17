import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import harmonicModelAnal as HA
import waveIO as WIO
import errorHandler as EH

try:
	import genSpecSines_C as GS
except ImportError:
	import genSpecSines as GS
	EH.printWarning(1)

def harmonicModelSynth(hfreq, hmag, hphase, N, H, fs):
	# Synthesis of a sound using the sinusoidal harmonic model
	# hfreq, hmag, hphase: harmonic frequencies, magnitudes and phases
	# returns y: output array sound
	
	hN = N/2      
	pin = 0                                                 # initialize output sound pointer 
	L = hfreq[:,0].size                                     # number of frames   
	ysize = H*(L+3)                                         # output sound size
	y = np.zeros(ysize)                                     # initialize output array
	sw = np.zeros(N)                                        # initialize synthesis window
	ow = triang(2*H)                                        # overlapping window
	sw[hN-H:hN+H] = ow      
	bh = blackmanharris(N)                                  # synthesis window
	bh = bh / sum(bh)                                       # normalize synthesis window
	sw[hN-H:hN+H] = sw[hN-H:hN+H] / bh[hN-H:hN+H]           # window for overlap-add
	for l in range(L): 
		Yh = GS.genSpecSines(N*hfreq[l,:]/fs, hmag[l,:], hphase[l,:], N)   # generate spec sines          
		yh = np.real(fftshift(ifft(Yh)))                      # inverse FFT
		y[pin:pin+N] += sw*yh                                 # overlap-add
		pin += H                                             # advance sound pointer
	return y

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

	hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
	y = harmonicModelSynth(hfreq, hmag, hphase, Ns, H, fs)
	WIO.play(y, fs)

