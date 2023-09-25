# function to call the main analysis/synthesis functions in software/models/stochasticModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal.windows import hann
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import stochasticModel as STM
import stft as STFT

def main(inputFile='../../sounds/ocean.wav', H=256, N=512, stocf=.1, melScale=1, normalization=1):
	"""
	inputFile: input sound file (monophonic with sampling rate of 44100)
	H: hop size, N: fft size
	stocf: decimation factor used for the stochastic approximation (bigger than 0, maximum 1)
	melScale: frequency approximation scale (0: linear approximation, 1: mel frequency approximation)
	normalization: amplitude normalization of output (0: no normalization, 1: normalization to input amplitude)
	"""

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute stochastic model
	stocEnv = STM.stochasticModelAnal(x, H, N, stocf, fs, melScale)

	# synthesize sound from stochastic model
	y = STM.stochasticModelSynth(stocEnv, H, N, fs, melScale)

	if (normalization==1):
		y = y * max(x)/max(y)

	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stochasticModel.wav'

	# write output sound
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(9, 6))

	# frequency range to plot
	maxplotfreq = 10000.0

	# plot input spectrogram
	plt.subplot(2,1,1)
	mX, pX = STFT.stftAnal(x, hann(N), N, H)
	numFrames = int(mX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/fs+1)]))
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('input magnitude spectrogram')
	plt.autoscale(tight=True)

	# plot the output sound
	plt.subplot(2,1,2)
	mY, pY = STFT.stftAnal(y, hann(N), N, H)
	numFrames = int(mY[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:,:int(N*maxplotfreq/fs+1)]))
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('input magnitude spectrogram')
	plt.autoscale(tight=True)

	plt.tight_layout()
	plt.ion()
	plt.show()

if __name__ == "__main__":
	main()
