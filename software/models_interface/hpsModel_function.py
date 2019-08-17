# function to call the main analysis/synthesis functions in software/models/hpsModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import hpsModel as HPS

def main(inputFile='../../sounds/sax-phrase-short.wav', window='blackman', M=601, N=1024, t=-100,
	minSineDur=0.1, nH=100, minf0=350, maxf0=700, f0et=5, harmDevSlope=0.01, stocf=0.1):
	"""
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
	M: analysis window size; N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
	nH: maximum number of harmonics; minf0: minimum fundamental frequency in sound
	maxf0: maximum fundamental frequency in sound; f0et: maximum error accepted in f0 detection algorithm
	harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
	stocf: decimation factor used for the stochastic approximation
	"""

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# compute the harmonic plus stochastic model of the whole sound
	hfreq, hmag, hphase, stocEnv = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)

	# synthesize a sound from the harmonic plus stochastic representation
	y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, stocEnv, Ns, H, fs)

	# output sound file (monophonic with sampling rate of 44100)
	outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel_sines.wav'
	outputFileStochastic = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel_stochastic.wav'
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel.wav'

	# write sounds files for harmonics, stochastic, and the sum
	UF.wavwrite(yh, fs, outputFileSines)
	UF.wavwrite(yst, fs, outputFileStochastic)
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(9, 6))

	# frequency range to plot
	maxplotfreq = 15000.0

	# plot the input sound
	plt.subplot(3,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot spectrogram stochastic component
	plt.subplot(3,1,2)
	numFrames = int(stocEnv[:,0].size)
	sizeEnv = int(stocEnv[0,:].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv
	plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv[:,:int(sizeEnv*maxplotfreq/(.5*fs)+1)]))
	plt.autoscale(tight=True)

	# plot harmonic on top of stochastic spectrogram
	if (hfreq.shape[1] > 0):
		harms = hfreq*np.less(hfreq,maxplotfreq)
		harms[harms==0] = np.nan
		numFrames = harms.shape[0]
		frmTime = H*np.arange(numFrames)/float(fs)
		plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
		plt.xlabel('time (sec)')
		plt.ylabel('frequency (Hz)')
		plt.autoscale(tight=True)
		plt.title('harmonics + stochastic spectrogram')

	# plot the output sound
	plt.subplot(3,1,3)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('output sound: y')

	plt.tight_layout()
	plt.ion()
	plt.show()

if __name__ == "__main__":
	main()
