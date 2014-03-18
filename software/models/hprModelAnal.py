import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))

import stftAnal as STFT
import waveIO as WIO
import harmonicModelAnal as HA
import sineSubtraction as SS

try:
	import genSpecSines_C as GS
except ImportError:
	import genSpecSines as GS
	EH.printWarning(1)
	

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
	hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
	xr = SS.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
	mXr, pXr = STFT.stftAnal(xr, fs, hamming(Ns), Ns, H)

	maxplotfreq = 20000.0
	numFrames = int(mXr[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = fs*np.arange(Ns*maxplotfreq/fs)/Ns                        
	plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:,:Ns*maxplotfreq/fs+1]))
	plt.autoscale(tight=True)

	harms = hfreq*np.less(hfreq,maxplotfreq)
	harms[harms==0] = np.nan
	numFrames = int(harms[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs) 
	plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
	plt.autoscale(tight=True)
	plt.title('harmonic + residual components')
	plt.show()
