# function to call the main analysis/synthesis functions in software/models/harmonicModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import sineModel as SM
import stft as STFT
import harmonicModel as HM

def main(inputFile='../sounds/vignesh.wav', window='blackman', M=1201, N=2048, t=-90, 
	minSineDur=0.1, nH=100, minf0=130, maxf0=300, f0et=7, harmDevSlope=0.01):
	# analysis and synthesis using the harmonic model
	# inputFile: input sound file (monophonic with sampling rate of 44100)
	# window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
	# M: analysis window size 
	# N: fft size (power of two, bigger or equal than M)
	# t: magnitude threshold of spectral peaks 
	# minSineDur: minimum duration of sinusoidal tracks
	# nH: maximum number of harmonics
	# minf0: minimum fundamental frequency in sound
	# maxf0: maximum fundamental frequency in sound
	# f0et: maximum error accepted in f0 detection algorithm                                                                                            
	# harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# --------- computation -----------------

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# compute spectrogram of input sound
	mX, pX = STFT.stftAnal(x, fs, w, N, H)

	# computer harmonics of input sound
	hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

	# synthesize harmonics
	y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)  

	# output sound file (monophonic with sampling rate of 44100)
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_harmonicModel.wav'

	# write the sound resulting from the inverse stft
	UF.wavwrite(y, fs, outputFile)

	# --------- plotting --------------------

	# create figure to show plots
	plt.figure(figsize=(12, 9))

	# frequency range to plot
	maxplotfreq = 5000.0

	# plot the input sound
	plt.subplot(3,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot magnitude spectrogram
	plt.subplot(3,1,2)
	numFrames = int(mX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N  
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
	plt.autoscale(tight=True)
	  
	# plot harmonics on top of spectrogram of input sound
	harms = hfreq*np.less(hfreq,maxplotfreq)
	harms[harms==0] = np.nan
	numFrames = int(hfreq[:,0].size)
	plt.plot(frmTime, harms, color='k')
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('magnitude spectrogram + harmonic tracks')
	plt.autoscale(tight=True)

	# plot the output sound
	plt.subplot(3,1,3)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('output sound: y')

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	main()