# function to call the main analysis/synthesis functions in software/models/stochasticModel.py

import numpy as np
import matplotlib.pyplot as plt
import os, sys
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import stochasticModel as STM

def main(inputFile='../../sounds/ocean.wav', H=256, N=512, stocf=.1):
	"""
	inputFile: input sound file (monophonic with sampling rate of 44100)
	H: hop size, N: fft size
	stocf: decimation factor used for the stochastic approximation (bigger than 0, maximum 1)
	"""

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# compute stochastic model
	stocEnv = STM.stochasticModelAnal(x, H, N, stocf)

	# synthesize sound from stochastic model
	y = STM.stochasticModelSynth(stocEnv, H, N)

	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stochasticModel.wav'

	# write output sound
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(9, 6))

	# plot the input sound
	plt.subplot(3,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot stochastic representation
	plt.subplot(3,1,2)
	numFrames = int(stocEnv[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = np.arange(stocf*(N/2+1))*float(fs)/(stocf*N)                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv))
	plt.autoscale(tight=True)
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('stochastic approximation')

	# plot the output sound
	plt.subplot(3,1,3)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')

	plt.tight_layout()
	plt.ion()
	plt.show()

if __name__ == "__main__":
	main()
