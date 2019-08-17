# function to call the main analysis/synthesis functions in software/models/sprModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import sprModel as SPR
import stft as STFT

def main(inputFile='../../sounds/bendir.wav', window='hamming', M=2001, N=2048, t=-80,
	minSineDur=0.02, maxnSines=150, freqDevOffset=10, freqDevSlope=0.001):
	"""
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
	M: analysis window size
	N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks
	minSineDur: minimum duration of sinusoidal tracks
	maxnSines: maximum number of parallel sinusoids
	freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
	freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
	"""

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# perform sinusoidal plus residual analysis
	tfreq, tmag, tphase, xr = SPR.sprModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope)

	# compute spectrogram of residual
	mXr, pXr = STFT.stftAnal(xr, w, N, H)

	# sum sinusoids and residual
	y, ys = SPR.sprModelSynth(tfreq, tmag, tphase, xr, Ns, H, fs)

	# output sound file (monophonic with sampling rate of 44100)
	outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sprModel_sines.wav'
	outputFileResidual = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sprModel_residual.wav'
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sprModel.wav'

	# write sounds files for sinusoidal, residual, and the sum
	UF.wavwrite(ys, fs, outputFileSines)
	UF.wavwrite(xr, fs, outputFileResidual)
	UF.wavwrite(y, fs, outputFile)

	# create figure to show plots
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

	# plot the sinusoidal frequencies on top of the residual spectrogram
	if (tfreq.shape[1] > 0):
		tracks = tfreq*np.less(tfreq, maxplotfreq)
		tracks[tracks<=0] = np.nan
		plt.plot(frmTime, tracks, color='k')
		plt.title('sinusoidal tracks + residual spectrogram')
		plt.autoscale(tight=True)

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
