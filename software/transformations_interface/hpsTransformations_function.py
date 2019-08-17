# function call to the transformation functions of relevance for the hpsModel

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
import hpsModel as HPS
import hpsTransformations as HPST
import harmonicTransformations as HT
import utilFunctions as UF

def analysis(inputFile='../../sounds/sax-phrase-short.wav', window='blackman', M=601, N=1024, t=-100, 
	minSineDur=0.1, nH=100, minf0=350, maxf0=700, f0et=5, harmDevSlope=0.01, stocf=0.1):
	"""
	Analyze a sound with the harmonic plus stochastic model
	inputFile: input sound file (monophonic with sampling rate of 44100)
	window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
	M: analysis window size 
	N: fft size (power of two, bigger or equal than M)
	t: magnitude threshold of spectral peaks 
	minSineDur: minimum duration of sinusoidal tracks
	nH: maximum number of harmonics
	minf0: minimum fundamental frequency in sound
	maxf0: maximum fundamental frequency in sound
	f0et: maximum error accepted in f0 detection algorithm                                                                                            
	harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
	stocf: decimation factor used for the stochastic approximation
	returns inputFile: input file name; fs: sampling rate of input file,
	        hfreq, hmag: harmonic frequencies, magnitude; mYst: stochastic residual
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
	hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)

	# synthesize the harmonic plus stochastic model without original phases
	y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, np.array([]), mYst, Ns, H, fs)

	# write output sound 
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModel.wav'
	UF.wavwrite(y,fs, outputFile)

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

	# plot spectrogram stochastic compoment
	plt.subplot(3,1,2)
	numFrames = int(mYst[:,0].size)
	sizeEnv = int(mYst[0,:].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:int(sizeEnv*maxplotfreq/(.5*fs))+1]))
	plt.autoscale(tight=True)

	# plot harmonic on top of stochastic spectrogram
	if (hfreq.shape[1] > 0):
		harms = hfreq*np.less(hfreq,maxplotfreq)
		harms[harms==0] = np.nan
		numFrames = int(harms[:,0].size)
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
	plt.show(block=False)

	return inputFile, fs, hfreq, hmag, mYst


def transformation_synthesis(inputFile, fs, hfreq, hmag, mYst, freqScaling = np.array([0, 1.2, 2.01, 1.2, 2.679, .7, 3.146, .7]), 
	freqStretching = np.array([0, 1, 2.01, 1, 2.679, 1.5, 3.146, 1.5]), timbrePreservation = 1, 
	timeScaling = np.array([0, 0, 2.138, 2.138-1.0, 3.146, 3.146])):
	"""
	transform the analysis values returned by the analysis function and synthesize the sound
	inputFile: name of input file
	fs: sampling rate of input file	
	hfreq, hmag: harmonic frequencies and magnitudes
	mYst: stochastic residual
	freqScaling: frequency scaling factors, in time-value pairs (value of 1 no scaling)
	freqStretching: frequency stretching factors, in time-value pairs (value of 1 no stretching)
	timbrePreservation: 1 preserves original timbre, 0 it does not
	timeScaling: time scaling factors, in time-value pairs
	"""
	
	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128
	
	# frequency scaling of the harmonics 
	hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

	# time scaling the sound
	yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreqt, hmagt, mYst, timeScaling)

	# synthesis from the trasformed hps representation 
	y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)

	# write output sound 
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_hpsModelTransformation.wav'
	UF.wavwrite(y,fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(12, 6))

	# frequency range to plot
	maxplotfreq = 15000.0

	# plot spectrogram of transformed stochastic compoment
	plt.subplot(2,1,1)
	numFrames = int(ystocEnv[:,0].size)
	sizeEnv = int(ystocEnv[0,:].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(ystocEnv[:,:int(sizeEnv*maxplotfreq/(.5*fs))+1]))
	plt.autoscale(tight=True)

	# plot transformed harmonic on top of stochastic spectrogram
	if (yhfreq.shape[1] > 0):
		harms = yhfreq*np.less(yhfreq,maxplotfreq)
		harms[harms==0] = np.nan
		numFrames = int(harms[:,0].size)
		frmTime = H*np.arange(numFrames)/float(fs) 
		plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
		plt.xlabel('time (sec)')
		plt.ylabel('frequency (Hz)')
		plt.autoscale(tight=True)
		plt.title('harmonics + stochastic spectrogram')

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
	inputFile, fs, hfreq, hmag, mYst = analysis()

	# transformation and synthesis
	transformation_synthesis(inputFile, fs, hfreq, hmag, mYst)

	plt.show()
