import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time
from scipy.fftpack import fft, ifft

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF


def stochasticModelFrame(x, w, N, stocf) :
	# x: input array sound, w: analysis window, N: FFT size,  
	# stocf: decimation factor of mag spectrum for stochastic analysis
	hN = N/2+1                                               # size of positive spectrum
	hM = (w.size)/2                                          # half analysis window size
	pin = hM                                                 # initialize sound pointer in middle of analysis window       
	fftbuffer = np.zeros(N)                                  # initialize buffer for FFT
	yw = np.zeros(w.size)                                    # initialize output sound frame
	w = w / sum(w)                                           # normalize analysis window
	#-----analysis-----             
	xw = x[pin-hM:pin+hM] * w                              # window the input sound
	X = fft(xw)                                            # compute FFT
	mX = 20 * np.log10( abs(X[:hN]) )                      # magnitude spectrum of positive frequencies
	mXenv = resample(np.maximum(-200, mX), mX.size*stocf)  # decimate the mag spectrum     
	pX = np.angle(X[:hN])
	#-----synthesis-----
	mY = resample(mXenv, hN)                               # interpolate to original size
	pY = 2*np.pi*np.random.rand(hN)                        # generate phase random values
	Y = np.zeros(N, dtype = complex)
	Y[:hN] = 10**(mY/20) * np.exp(1j*pY)                   # generate positive freq.
	Y[hN:] = 10**(mY[-2:0:-1]/20) * np.exp(-1j*pY[-2:0:-1]) # generate negative freq.
	fftbuffer = np.real( ifft(Y) )                         # inverse FFT
	y = fftbuffer*N/2                                  
	return mX, pX, mY, pY, y
    
# example call of stochasticModel function
if __name__ == '__main__':
  (fs, x) = UF.wavread('../../../sounds/ocean.wav')
  w = np.hanning(1024)
  N = 1024
  stocf = 0.1
  maxFreq = 10000.0
  lastbin = N*maxFreq/fs
  first = 1000
  last = first+w.size
  mX, pX, mY, pY, y = stochasticModelFrame(x[first:last], w, N, stocf)
  
  plt.figure(1, figsize=(9, 5))
  plt.subplot(3,1,1)
  plt.plot(float(fs)*np.arange(mY.size)/N, mY, 'r', lw=1.5, label="mY")
  plt.axis([0, maxFreq, -78, max(mX)+0.5])
  plt.title('mY (stochastic approximation of mX)')
  plt.subplot(3,1,2)
  plt.plot(float(fs)*np.arange(pY.size)/N, pY-np.pi, 'c', lw=1.5, label="pY")
  plt.axis([0, maxFreq, -np.pi, np.pi]) 
  plt.title('pY (random phases)')
  plt.subplot(3,1,3)
  plt.plot(np.arange(first, last)/float(fs), y, 'b', lw=1.5)
  plt.axis([first/float(fs), last/float(fs), min(y), max(y)])
  plt.title('yst')

  plt.tight_layout()
  plt.savefig('stochasticSynthesisFrame.png')
  plt.show()
