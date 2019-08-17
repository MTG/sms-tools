# function to call the main analysis/synthesis functions in software/models/harmonicModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import sineModel as SM
import harmonicModel as HM

def main(inputFile='../../sounds/vignesh.wav', window='blackman', M=1201, N=2048, t=-90,
	minSineDur=0.1, nH=100, minf0=130, maxf0=300, f0et=7, harmDevSlope=0.01):
	"""
	Analysis and synthesis using the harmonic model
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)
	M: analysis window size; N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
	nH: maximum number of harmonics; minf0: minimum fundamental frequency in sound
	maxf0: maximum fundamental frequency in sound; f0et: maximum error accepted in f0 detection algorithm
	harmDevSlope: allowed deviation of harmonic tracks, higher harmonics could have higher allowed deviation
	"""

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# detect harmonics of input sound
	hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

	# synthesize the harmonics
	y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)

	# output sound file (monophonic with sampling rate of 44100)
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_harmonicModel.wav'

	# write the sound resulting from harmonic analysis
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

	# plot the harmonic frequencies
	plt.subplot(3,1,2)
	if (hfreq.shape[1] > 0):
		numFrames = hfreq.shape[0]
		frmTime = H*np.arange(numFrames)/float(fs)
		hfreq[hfreq<=0] = np.nan
		plt.plot(frmTime, hfreq)
		plt.axis([0, x.size/float(fs), 0, maxplotfreq])
		plt.title('frequencies of harmonic tracks')

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
