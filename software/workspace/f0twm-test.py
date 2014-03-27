import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import sys
import math

sys.path.append('../models/')

import utilFunctions as UF
import stft as STFT
import sineModel as SM
import harmonicModel as HM

if __name__ == '__main__':
	(fs, x) = UF.wavread('../../sounds/vignesh.wav')
	w = np.blackman(1001)
	N = 1024
	t = -90
	minf0 = 100
	maxf0 = 300
	f0et = 5
	H= 256

	mX, pX = STFT.stftAnal(x, fs, w, N, H)
	f0 = HM.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et)
	freqs = np.zeros([f0.size,2])
	freqs[:,0] = f0
	y = SM.sineModelSynth(freqs, np.zeros([f0.size,2])-15, np.array([]), 1024, 256, fs)
	UF.play(y,fs)

	maxplotfreq = 5000.0
	numFrames = int(mX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
	plt.autoscale(tight=True)
	
	f0[f0==0] = np.nan
	plt.plot(frmTime, f0, linewidth=2, color='k')
	plt.autoscale(tight=True)
	plt.title('f0 on spectrogram')
	plt.show()

