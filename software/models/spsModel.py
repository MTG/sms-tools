# functions that implement analysis and synthesis of sounds using the Sinusoidal plus Stochastic Model
# (for example usage check the models_interface directory)

import numpy as np
from scipy.signal import resample
from scipy.signal.windows import blackmanharris, triang, hann
from scipy.fft import fft, ifft
import math
import utilFunctions as UF
import dftModel as DFT
import sineModel as SM
import stochasticModel as STM

def spsModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope, stocf):
	"""
	Analysis of a sound using the sinusoidal plus stochastic model
	x: input sound, fs: sampling rate, w: analysis window; N: FFT size, t: threshold in negative dB, 
	minSineDur: minimum duration of sinusoidal tracks
	maxnSines: maximum number of parallel sinusoids
	freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0   
	freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
	stocf: decimation factor used for the stochastic approximation
	returns hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; stocEnv: stochastic residual
	"""

	# perform sinusoidal analysis
	tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
	Ns = 512
	xr = UF.sineSubtraction(x, Ns, H, tfreq, tmag, tphase, fs)    	# subtract sinusoids from original sound
	stocEnv = STM.stochasticModelAnal(xr, H, H*2, stocf)            # compute stochastic model of residual
	return tfreq, tmag, tphase, stocEnv

def spsModelSynth(tfreq, tmag, tphase, stocEnv, N, H, fs):
	"""
	Synthesis of a sound using the sinusoidal plus stochastic model
	tfreq, tmag, tphase: sinusoidal frequencies, amplitudes and phases; stocEnv: stochastic envelope
	N: synthesis FFT size; H: hop size, fs: sampling rate 
	returns y: output sound, ys: sinusoidal component, yst: stochastic component
	"""

	ys = SM.sineModelSynth(tfreq, tmag, tphase, N, H, fs)          # synthesize sinusoids
	yst = STM.stochasticModelSynth(stocEnv, H, H*2)                # synthesize stochastic residual
	y = ys[:min(ys.size, yst.size)]+yst[:min(ys.size, yst.size)]   # sum sinusoids and stochastic components
	return y, ys, yst

	
def spsModel(x, fs, w, N, t, stocf):
	"""
	Analysis/synthesis of a sound using the sinusoidal plus stochastic model
	x: input sound, fs: sampling rate, w: analysis window, 
	N: FFT size (minimum 512), t: threshold in negative dB, 
	stocf: decimation factor of mag spectrum for stochastic analysis
	returns y: output sound, ys: sinusoidal component, yst: stochastic component
	"""

	hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
	Ns = 512                                                      # FFT size for synthesis (even)
	H = Ns//4                                                     # Hop size used for analysis and synthesis
	hNs = Ns//2      
	pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
	pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
	ysw = np.zeros(Ns)                                            # initialize output sound frame
	ystw = np.zeros(Ns)                                           # initialize output sound frame
	ys = np.zeros(x.size)                                         # initialize output array
	yst = np.zeros(x.size)                                        # initialize output array
	w = w / sum(w)                                                # normalize analysis window
	sw = np.zeros(Ns)     
	ow = triang(2*H)                                              # overlapping window
	sw[hNs-H:hNs+H] = ow      
	bh = blackmanharris(Ns)                                       # synthesis window
	bh = bh / sum(bh)                                             # normalize synthesis window
	wr = bh                                                       # window for residual
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
	sws = H*hann(Ns)/2                                         # synthesis window for stochastic

	while pin<pend:  
	#-----analysis-----             
		x1 = x[pin-hM1:pin+hM2]                                     # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                              # compute dft
		ploc = UF.peakDetection(mX, t)                              # find peaks 
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)         # refine peak values		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)          # refine peak values
		ipfreq = fs*iploc/float(N)                                  # convert peak locations to Hertz
		ri = pin-hNs-1                                              # input sound pointer for residual analysis
		xw2 = x[ri:ri+Ns]*wr                                        # window the input sound                                       
		fftbuffer = np.zeros(Ns)                                    # reset buffer
		fftbuffer[:hNs] = xw2[hNs:]                                 # zero-phase window in fftbuffer
		fftbuffer[hNs:] = xw2[:hNs]                           
		X2 = fft(fftbuffer)                                         # compute FFT for residual analysis
		
	#-----synthesis-----
		Ys = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)        # generate spec of sinusoidal component          
		Xr = X2-Ys                                                  # get the residual complex spectrum
		mXr = 20 * np.log10(abs(Xr[:hNs]))                          # magnitude spectrum of residual
		mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)    # decimate the magnitude spectrum and avoid -Inf                     
		stocEnv = resample(mXrenv, hNs)                             # interpolate to original size
		pYst = 2*np.pi*np.random.rand(hNs)                          # generate phase random values
		Yst = np.zeros(Ns, dtype = complex)
		Yst[:hNs] = 10**(stocEnv/20) * np.exp(1j*pYst)              # generate positive freq.
		Yst[hNs+1:] = 10**(stocEnv[:0:-1]/20) * np.exp(-1j*pYst[:0:-1])  # generate negative freq.

		fftbuffer = np.real(ifft(Ys))                               # inverse FFT of harmonic spectrum
		ysw[:hNs-1] = fftbuffer[hNs+1:]                             # undo zero-phase window
		ysw[hNs-1:] = fftbuffer[:hNs+1] 

		fftbuffer = np.real(ifft(Yst))                              # inverse FFT of stochastic spectrum
		ystw[:hNs-1] = fftbuffer[hNs+1:]                            # undo zero-phase window
		ystw[hNs-1:] = fftbuffer[:hNs+1]

		ys[ri:ri+Ns] += sw*ysw                                      # overlap-add for sines
		yst[ri:ri+Ns] += sws*ystw                                   # overlap-add for stochastic
		pin += H                                                    # advance sound pointer
			
	y = ys+yst                                                    # sum of sinusoidal and residual components
	return y, ys, yst

