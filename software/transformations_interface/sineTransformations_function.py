# function call to the transformation functions of relevance for the sineModel

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
import sineModel as SM
import sineTransformations as ST
import utilFunctions as UF

def analysis(inputFile='../../sounds/mridangam.wav', window='hamming', M=801, N=2048, t=-90, 
	minSineDur=0.01, maxnSines=150, freqDevOffset=20, freqDevSlope=0.02):
	"""
	Analyze a sound with the sine model
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
	M: analysis window size; N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks; minSineDur: minimum duration of sinusoidal tracks
	maxnSines: maximum number of parallel sinusoids
	freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0   
	freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
	returns inputFile: input file name; fs: sampling rate of input file,
	        tfreq, tmag: sinusoidal frequencies and magnitudes
	"""

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# compute the sine model of the whole sound
	tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

	# synthesize the sines without original phases
	y = SM.sineModelSynth(tfreq, tmag, np.array([]), Ns, H, fs)

	# output sound file (monophonic with sampling rate of 44100)
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModel.wav'

	# write the sound resulting from the inverse stft
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
		
	# plot the sinusoidal frequencies
	if (tfreq.shape[1] > 0):
		plt.subplot(3,1,2)
		tracks = np.copy(tfreq)
		tracks = tracks*np.less(tracks, maxplotfreq)
		tracks[tracks<=0] = np.nan
		numFrames = int(tracks[:,0].size)
		frmTime = H*np.arange(numFrames)/float(fs)
		plt.plot(frmTime, tracks)
		plt.axis([0, x.size/float(fs), 0, maxplotfreq])
		plt.title('frequencies of sinusoidal tracks')

	# plot the output sound
	plt.subplot(3,1,3)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('output sound: y')

	plt.tight_layout()
	plt.show(block=False)

	return inputFile, fs, tfreq, tmag


def transformation_synthesis(inputFile, fs, tfreq, tmag, freqScaling = np.array([0, 2.0, 1, .3]), 
	timeScaling = np.array([0, .0, .671, .671, 1.978, 1.978+1.0])):
	"""
	Transform the analysis values returned by the analysis function and synthesize the sound
	inputFile: name of input file; fs: sampling rate of input file	
	tfreq, tmag: sinusoidal frequencies and magnitudes
	freqScaling: frequency scaling factors, in time-value pairs
	timeScaling: time scaling factors, in time-value pairs
	"""

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# frequency scaling of the sinusoidal tracks 
	ytfreq = ST.sineFreqScaling(tfreq, freqScaling)

	# time scale the sinusoidal tracks 
	ytfreq, ytmag = ST.sineTimeScaling(ytfreq, tmag, timeScaling)

	# synthesis 
	y = SM.sineModelSynth(ytfreq, ytmag, np.array([]), Ns, H, fs)

	# write output sound 
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModelTransformation.wav'
	UF.wavwrite(y,fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(12, 6))

	# frequency range to plot
	maxplotfreq = 15000.0

	# plot the transformed sinusoidal frequencies
	if (ytfreq.shape[1] > 0):
		plt.subplot(2,1,1)
		tracks = np.copy(ytfreq)
		tracks = tracks*np.less(tracks, maxplotfreq)
		tracks[tracks<=0] = np.nan
		numFrames = int(tracks[:,0].size)
		frmTime = H*np.arange(numFrames)/float(fs)
		plt.plot(frmTime, tracks)
		plt.title('transformed sinusoidal tracks')
		plt.autoscale(tight=True)

	# plot the output sound
	plt.subplot(2,1,2)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('output sound: y')

	plt.tight_layout()
	plt.show()

if __name__ == "__main__":
	
	# analysis
	inputFile, fs, tfreq, tmag = analysis()

	# transformation and synthesis
	transformation_synthesis (inputFile, fs, tfreq, tmag)

	plt.show()

