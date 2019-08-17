# function to call the main analysis/synthesis functions in software/models/spsModel.py

import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import spsModel as SPS
import utilFunctions as UF

def main(inputFile='../../sounds/bendir.wav', window='hamming', M=2001, N=2048, t=-80, minSineDur=0.02,
	maxnSines=150, freqDevOffset=10, freqDevSlope=0.001, stocf=0.2):
	"""
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
	M: analysis window size; N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
	maxnSines: maximum number of parallel sinusoids
	freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
	freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
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

	# perform sinusoidal+sotchastic analysis
	tfreq, tmag, tphase, stocEnv = SPS.spsModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, stocf)

	# synthesize sinusoidal+stochastic model
	y, ys, yst = SPS.spsModelSynth(tfreq, tmag, tphase, stocEnv, Ns, H, fs)

	# output sound file (monophonic with sampling rate of 44100)
	outputFileSines = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_spsModel_sines.wav'
	outputFileStochastic = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_spsModel_stochastic.wav'
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_spsModel.wav'

	# write sounds files for sinusoidal, residual, and the sum
	UF.wavwrite(ys, fs, outputFileSines)
	UF.wavwrite(yst, fs, outputFileStochastic)
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(9, 6))

	# frequency range to plot
	maxplotfreq = 10000.0

	# plot the input sound
	plt.subplot(3,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	plt.subplot(3,1,2)
	numFrames = int(stocEnv[:,0].size)
	sizeEnv = int(stocEnv[0,:].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv
	plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv[:,:int(sizeEnv*maxplotfreq/(.5*fs)+1)]))
	plt.autoscale(tight=True)

	# plot sinusoidal frequencies on top of stochastic component
	if (tfreq.shape[1] > 0):
		sines = tfreq*np.less(tfreq,maxplotfreq)
		sines[sines==0] = np.nan
		numFrames = int(sines[:,0].size)
		frmTime = H*np.arange(numFrames)/float(fs)
		plt.plot(frmTime, sines, color='k', ms=3, alpha=1)
		plt.xlabel('time(s)')
		plt.ylabel('Frequency(Hz)')
		plt.autoscale(tight=True)
		plt.title('sinusoidal + stochastic spectrogram')

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
