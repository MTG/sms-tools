# function call to the transformation function of relevance to the stochasticModel

import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../transformations/'))
import stochasticModel as STC
import utilFunctions as UF
import stochasticTransformations as STCT

def main (inputFile='../../sounds/rain.wav', stocf=0.1, timeScaling = np.array([0, 0, 1, 2])):
	"""
	function to perform a time scaling using the stochastic model
	inputFile: name of input sound file
	stocf: decimation factor used for the stochastic approximation
	timeScaling: time scaling factors, in time-value pairs
	"""

	# hop size 
	H = 128

	# read input sound
	(fs, x) = UF.wavread(inputFile)

	# perform stochastic analysis
	mYst = STC.stochasticModelAnal(x, H, H*2, stocf)
	        
	# perform time scaling of stochastic representation
	ystocEnv = STCT.stochasticTimeScale(mYst, timeScaling)
	
	# synthesize output sound
	y = STC.stochasticModelSynth(ystocEnv, H, H*2)
	
	# write output sound
	outputFile = 'output_sounds/' + os.path.basename(inputFile)[:-4] + '_stochasticModelTransformation.wav'
	UF.wavwrite(y, fs, outputFile)

	# create figure to plot
	plt.figure(figsize=(9, 6))

	# plot the input sound
	plt.subplot(4,1,1)
	plt.plot(np.arange(x.size)/float(fs), x)
	plt.axis([0, x.size/float(fs), min(x), max(x)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')
	plt.title('input sound: x')

	# plot stochastic representation
	plt.subplot(4,1,2)
	numFrames = int(mYst[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = np.arange(int(stocf*H))*float(fs)/(stocf*2*H)                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(mYst))
	plt.autoscale(tight=True)
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('stochastic approximation')

	# plot modified stochastic representation
	plt.subplot(4,1,3)
	numFrames = int(ystocEnv[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = np.arange(int(stocf*H))*float(fs)/(stocf*2*H)                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(ystocEnv))
	plt.autoscale(tight=True)
	plt.xlabel('time (sec)')
	plt.ylabel('frequency (Hz)')
	plt.title('modified stochastic approximation')

	# plot the output sound
	plt.subplot(4,1,4)
	plt.plot(np.arange(y.size)/float(fs), y)
	plt.axis([0, y.size/float(fs), min(y), max(y)])
	plt.ylabel('amplitude')
	plt.xlabel('time (sec)')

	plt.tight_layout()
	plt.show()

if __name__ == '__main__':
	main()
