import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import utilFunctions as UF
import harmonicModel as HM
import stft as STFT

def harmonicPitchScale(hfreq, freqScaling, freqStretching):
	# frequency scaling of harmonics
	# hfreq: frequencies of input harmonics
	# freqScaling: scaling factors, in time-value pairs
	# freqStretching: stretching factors, in time-value pairs
	# returns hfreq: frequencies output harmonics
	L = hfreq[:,0].size            # number of frames
	freqScaling = np.interp(np.arange(L), L*freqScaling[::2]/freqScaling[-2], freqScaling[1::2]) 
	freqSretching = np.interp(np.arange(L), L*freqStretching[::2]/freqStretching[-2], freqStretching[1::2]) 
	yhfreq = np.empty_like(hfreq)  # create empty output matrix
	for l in range(L):             # go through all frames
		yhfreq[l,:] = sfreq[l,:] * freqScaling[l]
	return yhfreq

if __name__ == '__main__':
	(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/speech.wav'))
	w = np.hamming(1701)
	N = 2048
	t = -90
	nH = 100
	minf0 = 52
	maxf0 = 118
	f0et = 15
	maxnpeaksTwm = 4
	minSineDur = .1
	harmDevSlope = 0.01
	Ns = 512
	H = Ns/4
	# hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
	# xr = SS.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
	# freqScaling = np.array([0, 0.165, 0.595, 0.850, 1.15, 2.15, 2.81, 3.285, 4.585, 4.845, 5.1, 6.15, 6.825, 7.285, 8.185, 8.830])
	# freqStretching = np.array([0, 0.165, 0.595, 0.850, .9+1.15, .2+2.15, 2.81, 3.285, 4.585, .6+4.845, .4+5.1, 6.15, 6.825, 7.285, 8.185, 8.830])            
 #	hfreq = harmonicPitchScale(hfreq, freqScaling, freqStretching)
#	yh = HS.harmonicModelSynth(hfreq, hmag, np.array([]), Ns, H, fs) 
#	y = xr + yh   

	mX, pX = STFT.stftAnal(x, fs, w, N, H)
	hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
	xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
	yh = HM.harmonicModelSynth(hfreq, hmag, hphase, Ns, H, fs) 
	UF.play(xr, fs)

	plt.figure(1, figsize=(9.5, 7))    
	maxplotfreq = 500.0
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

	plt.tight_layout()
	plt.show()        



