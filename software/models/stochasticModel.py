# functions that implement analysis and synthesis of sounds using the Stochastic Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal import hanning, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import utilFunctions as UF

def stochasticModelAnal(x, H, N, stocf):
	# Stochastic analysis of a sound
	# x: input array sound, H: hop size, N: fftsize
	# stocf: decimation factor of mag spectrum for stochastic analysis, bigger than 0, maximum of 1
	# returns stocEnv: stochastic envelope
  
  	hN = N/2                                                  # half of fft size
	w = hanning(N)                                            # analysis window
	x = np.append(np.zeros(hN),x)                             # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hN))                             # add zeros at the end to analyze last sample
	pin = hN                                                  # initialize sound pointer in middle of analysis window       
	pend = x.size-hN                                          # last sample to start a frame
	while pin<=pend:                         
		xw = x[pin-hN:pin+hN] * w                             # window the input sound
		X = fft(xw)                                           # compute FFT
		mX = 20 * np.log10(abs(X[:hN]))                       # magnitude spectrum of positive frequencies
		mY = resample(np.maximum(-200, mX), hN*stocf)         # decimate the mag spectrum 
		if pin == hN:                                         # first frame
			stocEnv = np.array([mY])
		else:                                                 # rest of frames
			stocEnv = np.vstack((stocEnv, np.array([mY])))
		pin += H                                              # advance sound pointer
	return stocEnv

def stochasticModelSynth(stocEnv, H, N):
	# Stochastic synthesis of a sound
	# stocEnv: stochastic envelope; H: hop size; N: fft size
	# returns y: output sound
 
	hN = N/2                                                 # half of FFT size for synthesis
	L = stocEnv[:,0].size                                    # number of frames
	ysize = H*(L+3)                                          # output sound size
	y = np.zeros(ysize)                                      # initialize output array
	ws = 2*hanning(N)                                        # synthesis window
	pout = 0                                                 # output sound pointer
	for l in range(L):                    
		mY = resample(stocEnv[l,:], hN)                      # interpolate to original size
		pY = 2*np.pi*np.random.rand(hN)                      # generate phase random values
		Y = np.zeros(N, dtype = complex)                     # initialize synthesis spectrum
		Y[:hN] = 10**(mY/20) * np.exp(1j*pY)                 # generate positive freq.
		Y[hN+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1])  # generate negative freq.
		fftbuffer = np.real(ifft(Y))                         # inverse FFT
		y[pout:pout+N] += ws*fftbuffer                       # overlap-add
		pout += H  
	y = np.delete(y, range(hN))                              # delete half of first window
	y = np.delete(y, range(y.size-hN, y.size))               # delete half of the last window 
	return y

def stochasticModel(x, H, N, stocf):
	# Stochastic analysis/synthesis of a sound, one frame at a time
	# x: input array sound, H: hop size, N: fft size 
	# stocf: decimation factor of mag spectrum for stochastic analysis, bigger than 0, maximum of 1
	# returns y: output sound

	hN = N/2                                                 # half of FFT size for synthesis
	w = hanning(N)                                           # analysis/synthesis window
	x = np.append(np.zeros(hN),x)                            # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hN))                            # add zeros at the end to analyze last sample
	pin = hN                                                 # initialize sound pointer in middle of analysis window       
	pend = x.size - hN                                       # last sample to start a frame
	y = np.zeros(x.size)                                     # initialize output array
	while pin<=pend:              
	#-----analysis-----             
		xw = x[pin-hN:pin+hN]*w                              # window the input sound
		X = fft(xw)                                          # compute FFT
		mX = 20 * np.log10(abs(X[:hN]))                         # magnitude spectrum of positive frequencies
		stocEnv = resample(np.maximum(-200, mX), mX.size*stocf) # decimate the mag spectrum     
	#-----synthesis-----
		mY = resample(stocEnv, hN)                            # interpolate to original size
		pY = 2*np.pi*np.random.rand(hN)                       # generate phase random values
		Y = np.zeros(N, dtype = complex)
		Y[:hN] = 10**(mY/20) * np.exp(1j*pY)                  # generate positive freq.
		Y[hN+1:] = 10**(mY[:0:-1]/20) * np.exp(-1j*pY[:0:-1]) # generate negative freq.
		fftbuffer = np.real(ifft(Y))                          # inverse FFT
		y[pin-hN:pin+hN] += w*fftbuffer                       # overlap-add
		pin += H  
	y = np.delete(y, range(hN))                               # delete half of first window which was added 
	y = np.delete(y, range(y.size-hN, y.size))                # delete half of last window which was added                                            # advance sound pointer
	return y
	
