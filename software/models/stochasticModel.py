import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hanning, resample
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time
import utilFunctions as UF
	
def stochasticModelAnal(x, H, stocf):
	# stochastic analysis of a sound
	# x: input array sound, H: hop size, 
	# stocf: decimation factor of mag spectrum for stochastic analysis
	# returns mYst: stochastic envelope
	N = H*2                                                  # FFT size   
	w = hanning(N)                                           # analysis window
	x = np.append(np.zeros(H),x)                             # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(H))                             # add zeros at the end to analyze last sample
	pin = 0                                                  # initialize sound pointer in middle of analysis window       
	pend = x.size-N                                          # last sample to start a frame
	y = np.zeros(x.size)                                     # initialize output array
	while pin<=pend:                         
		xw = x[pin:pin+N] * w                                  # window the input sound
		X = fft(xw)                                            # compute FFT
		mX = 20 * np.log10(abs(X[:H]))                         # magnitude spectrum of positive frequencies
		mY = resample(np.maximum(-200, mX), mX.size*stocf)     # decimate the mag spectrum 
		if pin == 0:
			mYst = np.array([mY])
		else:
			mYst = np.vstack((mYst, np.array([mY])))
		pin += H                                               # advance sound pointer
	return mYst

def stochasticModelSynth(mYst, H):
	# stochastic synthesis of a sound
	# mYst: stochastic envelope, H: hop size, 
	# returns y: output sound
	N = H*2                                                  # FFT size    
	L = mYst[:,0].size                                       # number of frames
	y = np.zeros(L*H+2*H)                                    # initialize output array
	ws = hanning(N)                                          # synthesis window
	pout = 0                                                 # output sound pointer
	for l in range(L):                    
		mY = resample(mYst[l,:], H)                            # interpolate to original size
		pY = 2*np.pi*np.random.rand(H)                         # generate phase random values
		Y = np.zeros(N, dtype = complex)
		Y[:H] = 10**(mY/20) * np.exp(1j*pY)                    # generate positive freq.
		Y[H+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1])   # generate negative freq.
		fftbuffer = np.real(ifft(Y))                           #inverse FFT
		y[pout:pout+N] += ws*fftbuffer                         # overlap-add
		pout += H  
	y = np.delete(y, range(H))                               # delete half of first window which was added 
	y = np.delete(y, range(y.size-H, y.size))                # delete half of last window which was added                                            # advance sound pointer
	return y

def stochasticModel(x, H, stocf):
	# stochastic analysis/synthesis of a sound, one frame at a time
	# x: input array sound, H: hop size, 
	# stocf: decimation factor of mag spectrum for stochastic analysis
	# returns y: output sound
	N = H*2                                                  # FFT size
	w = hanning(N)                                           # analysis/synthesis window
	x = np.append(np.zeros(H),x)                             # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(H))                             # add zeros at the end to analyze last sample
	pin = 0                                                  # initialize sound pointer in middle of analysis window       
	pend = x.size-N                                          # last sample to start a frame
	y = np.zeros(x.size)                                     # initialize output array
	while pin<=pend:              
	#-----analysis-----             
		xw = x[pin:pin+N]*w                                    # window the input sound
		X = fft(xw)                                            # compute FFT
		mX = 20 * np.log10(abs(X[:H]))                         # magnitude spectrum of positive frequencies
		mYst = resample(np.maximum(-200, mX), mX.size*stocf)   # decimate the mag spectrum     
	#-----synthesis-----
		mY = resample(mYst, H)                                 # interpolate to original size
		pY = 2*np.pi*np.random.rand(H)                         # generate phase random values
		Y = np.zeros(N, dtype = complex)
		Y[:H] = 10**(mY/20) * np.exp(1j*pY)                    # generate positive freq.
		Y[H+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1])   # generate negative freq.
		fftbuffer = np.real(ifft(Y))                           # inverse FFT
		y[pin:pin+N] += w*fftbuffer                            # overlap-add
		pin += H  
	y = np.delete(y, range(H))                               # delete half of first window which was added 
	y = np.delete(y, range(y.size-H, y.size))                # delete half of last window which was added                                            # advance sound pointer
	return y
		
		
# example use of the stochastic model functions
if __name__ == '__main__':
	(fs, x) = UF.wavread('../../sounds/ocean.wav')
	H = 256
	stocf = .1
	mYst = stochasticModelAnal(x, H, stocf)
	y = stochasticModelSynth(mYst, H)
	UF.wavwrite(y, fs, 'ocean-stochasticModel.wav')

	# plot stochastic representation
	plt.figure(1, figsize=(9.5, 7)) 
	numFrames = int(mYst[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = np.arange(stocf*H)*float(fs)/(stocf*2*H)                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(mYst))
	plt.autoscale(tight=True)
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
	plt.title('stochastic approximation')

	plt.tight_layout()
	plt.show()

