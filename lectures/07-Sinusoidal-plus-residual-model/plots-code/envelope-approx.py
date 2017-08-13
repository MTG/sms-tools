import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time
from scipy.fftpack import fft, ifft

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF
import dftModel as DFT


def stochasticModelFrame(x, w, N, stocf) :
	# x: input array sound, w: analysis window, N: FFT size,  
	# stocf: decimation factor of mag spectrum for stochastic analysis
	hN = N//2+1                                               # size of positive spectrum
	hM = (w.size)//2                                          # half analysis window size
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
  stocf = .1
  envSize = int((N*stocf)/2)
  maxFreq = 10000.0
  lastbin = int(N*maxFreq/fs)
  first = 4000
  last = first+w.size
  mX, pX = DFT.dftAnal(x[first:last], w, N)
  mXenv = resample(np.maximum(-200, mX), envSize)
  mY = resample(mXenv, N//2)
 
  plt.figure(1, figsize=(9, 5))
  plt.plot(float(fs)*np.arange(mX.size)/N, mX, 'r', lw=1.5, label=r'$a$')
  plt.plot(float(fs/2.0)*np.arange(0, envSize)/envSize, mXenv, color='k', lw=1.5, label=r'$\tilde a$')
  plt.plot(float(fs)*np.arange(0, N/2)/N, mY, 'g', lw=1.5, label=r'$b$')
  plt.legend(fontsize=18)
  plt.axis([0, maxFreq, -75, max(mX)])
  plt.title('envelope approximation')

  plt.tight_layout()
  plt.savefig('envelope-approx.png')
  plt.show()
