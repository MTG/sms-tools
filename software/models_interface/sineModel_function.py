# function to call the main analysis/synthesis functions in software/models/sineModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import sineModel as SM
import stft as STFT

def main(inputFile='../../sounds/bendir.wav', window='hamming', M=2001, N=2048, t=-80, minSineDur=.02, maxnSines=150, freqDevOffset=10, freqDevSlope=0.001):

	# ------- analysis parameters -------------------

	# inputFile: input sound file (monophonic with sampling rate of 44100)
	# window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
	# M: analysis window size 
	# N: fft size (power of two, bigger or equal than M)
	# t: magnitude threshold of spectral peaks 
	# minSineDur: minimum duration of sinusoidal tracks
	# maxnSines: maximum number of parallel sinusoids
	# freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0   
	# freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# --------- computation -----------------

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# compute the magnitude and phase spectrogram of input sound
	mX, pX = STFT.stftAnal(x, fs, w, N, H)

	# compute the sinusoidal model
	tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

	# synthesize the output sound from the sinusoidal representation
	y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

	# output sound file (monophonic with sampling rate of 44100)
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_sineModel.wav'

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
		
	# plot the magnitude spectrogram
	plt.subplot(3,1,2)
	maxplotbin = int(N*maxplotfreq/fs)
	numFrames = int(mX[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                       
	binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
	plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
	plt.autoscale(tight=True)
		
	# plot the sinusoidal frequencies on top of the spectrogram
	tracks = tfreq*np.less(tfreq, maxplotfreq)
	tracks[tracks<=0] = np.nan
	plt.plot(frmTime, tracks, color='k')
	plt.title('magnitude spectrogram + sinusoidal tracks')
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

