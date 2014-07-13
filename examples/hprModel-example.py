# example of using the functions in software/models/hpsModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import sineModel as SM
import stft as STFT
import harmonicModel as HM

def main(inputFile='../sounds/sax-phrase.wav', window='blackman', M=601, N=1024, t=-100, 
	minSineDur=0.1, nH=100, minf0=350, maxf0=700, f0et=5, harmDevSlope=0.01):
	
	# ------- analysis parameters -------------------

	# inputFile: input sound file (monophonic with sampling rate of 44100)
	# window: analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
	# M: analysis window size 
	# N: fft size (power of two, bigger or equal than M)
	# t: magnitude threshold of spectral peaks 
	# minSineDur: minimum duration of sinusoidal tracks
	# nH: maximum number of harmonics
	# minf0: minimum fundamental frequency in sound
	# maxf0: maximum fundamental frequency in sound
	# f0et: maximum error accepted in f0 detection algorithm                                                                                            
	# harmDevSlope: allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation

	# size of fft used in synthesis
	Ns = 512

	# hop size (has to be 1/4 of Ns)
	H = 128

	# --------- computation -----------------

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute analysis window
	w = get_window(window, M)

	# find harmonics
	hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
	  
	# subtract harmonics from original sound
	xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
	  
	# compute spectrogram of residual
	mXr, pXr = STFT.stftAnal(xr, fs, w, N, H)
	  
	# synthesize harmonic component
	yh = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)

	# sum harmonics and residual
	y = xr[:min(xr.size, yh.size)]+yh[:min(xr.size, yh.size)]

	# output sound file (monophonic with sampling rate of 44100)
	outputFileSines = '../gui/output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_sines.wav'
	outputFileResidual = '../gui/output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel_residual.wav'
	outputFile = '../gui/output_sounds/' + os.path.basename(inputFile)[:-4] + '_hprModel.wav'

	# write sounds files for harmonics, residual, and the sum
	UF.wavwrite(yh, fs, outputFileSines)
	UF.wavwrite(xr, fs, outputFileResidual)
	UF.wavwrite(y, fs, outputFile)

	# --------- plotting --------------------

	# create figure to plot
	plt.figure(1, figsize=(12, 9))

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
	harms = hfreq*np.less(hfreq,maxplotfreq)
	harms[harms==0] = np.nan
	numFrames = int(harms[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs) 
	plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
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
	plt.show()

if __name__ == "__main__":
	main()


