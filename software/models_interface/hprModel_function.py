# function to call the main analysis/synthesis functions in software/models/hprModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import hprModel as HPR
import stft as STFT

def main(inputFile='../../sounds/sax-phrase-short.wav', window='blackman', M=601, N=1024, t=-100,
	minSineDur=0.1, nH=100, minf0=350, maxf0=700, f0et=5, harmDevSlope=0.01):
	"""
	Perform analysis/synthesis using the harmonic plus residual model
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
	M: analysis window size; N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
	nH: maximum number of harmonics; minf0: minimum fundamental frequency in sound
	maxf0: maximum fundamental frequency in sound; f0et: maximum error accepted in f0 detection algorithm
	harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
	"""

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# find harmonics and residual
	hfreq, hmag, hphase, xr = HPR.hprModelAnal(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope)

	# compute spectrogram of residual
	mXr, pXr = STFT.stftAnal(xr, w, N, H)
	  
	# synthesize hpr model
	y, yh = HPR.hprModelSynth(hfreq, hmag, hphase, xr, Ns, H, fs)

	# output sound file (monophonic with sampling rate of 44100)
	outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_sines.wav'
	outputFileResidual = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_residual.wav'
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel.wav'

	# write sounds files for harmonics, residual, and the sum
	UF.wavwrite(yh, fs, outputFileSines)
	UF.wavwrite(xr, fs, outputFileResidual)
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(9, 6))

	# frequency range to plot
	maxplotfreq = 5000.0

	# plot the input sound
	plt.subplot(3,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot the magnitude spectrogram of residual
	plt.subplot(3,1,2)
	maxplotbin = int(N*maxplotfreq/fs)
	numFrames = int(mXr[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = np.arange(maxplotbin+1)*float(fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:,:maxplotbin+1]))
	plt.autoscale(tight=True)

	# plot harmonic frequencies on residual spectrogram
	if (hfreq.shape[1] > 0):
		harms = hfreq*np.less(hfreq,maxplotfreq)
		harms[harms==0] = np.nan
		numFrames = int(harms[:,0].size)
		frmTime = H*np.arange(numFrames)/float(fs)
		plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
		plt.xlabel('time(s)')
		plt.ylabel('frequency(Hz)')
		plt.autoscale(tight=True)
		plt.title('harmonics + residual spectrogram')

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
