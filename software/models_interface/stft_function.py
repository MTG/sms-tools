# function to call the main analysis/synthesis functions in software/models/stft.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import stft as STFT

def main(inputFile = '../../sounds/piano.wav', window = 'hamming', M = 1024, N = 1024, H = 512):
	"""
	analysis/synthesis using the STFT
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris)
	M: analysis window size
	N: fft size (power of two, bigger or equal than M)
	H: hop size (at least 1/2 of analysis window size to have good overlap-add)
	"""

	# read input sound (monophonic with sampling rate of 44100)
	fs, x = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# compute the magnitude and phase spectrogram

	mX, pX = STFT.stftAnal(x, w, N, H)
	 
	# perform the inverse stft
	y = STFT.stftSynth(mX, pX, M, H)

	# output sound file (monophonic with sampling rate of 44100)
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stft.wav'

	# write the sound resulting from the inverse stft
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(9, 6))

	# frequency range to plot
	maxplotfreq = 5000.0

	# plot the input sound
	plt.subplot(4,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot magnitude spectrogram
	plt.subplot(4,1,2)
	numFrames = int(mX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/fs+1)]))
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('magnitude spectrogram')
	plt.autoscale(tight=True)

	# plot the phase spectrogram
	plt.subplot(4,1,3)
	numFrames = int(pX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = fs*np.arange(N*maxplotfreq/fs)/N
	plt.pcolormesh(frmTime, binFreq, np.transpose(np.diff(pX[:,:int(N*maxplotfreq/fs+1)],axis=1)))
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('phase spectrogram (derivative)')
	plt.autoscale(tight=True)

	# plot the output sound
	plt.subplot(4,1,4)
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
