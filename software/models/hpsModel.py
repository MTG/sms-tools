import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris, hanning, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
import harmonicModel as HM
import stft as STFT
import dftModel as DFT
import utilFunctions as UF


def hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf):
	# Analysis of a sound using the harmonic plus stochastic model
	# x: input sound, fs: sampling rate, w: analysis window, 
	# N: FFT size, t: threshold in negative dB, 
	# nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
	# maxf0: maximim f0 frequency in Hz, 
	# f0et: error threshold in the f0 detection (ex: 5),
	# harmDevSlope: slope of harmonic deviation
	# minSineDur: minimum length of harmonics
	# returns: hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; mYst: stochastic residual
	hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
	mYst = UF.stochasticResidual(x, Ns, H, hfreq, hmag, hphase, fs, stocf)
	return hfreq, hmag, hphase, mYst

def hpsModelSynth(hfreq, hmag, hphase, mYst, N, H, fs):
	# Synthesis of a sound using the harmonic plus stochastic model
	# hfreq: harmonic frequencies, hmag:harmonic amplitudes, mYst: stochastic envelope
	# Ns: synthesis FFT size, H: hop size, fs: sampling rate 
	# y: output sound, yh: harmonic component, yst: stochastic component
	hN = N/2                                                  # half of FFT size for synthesis
	L = hfreq[:,0].size                                       # number of frames
	nH = hfreq[0,:].size                                      # number of harmonics
	pout = 0                                                  # initialize output sound pointer         
	ysize = H*(L+4)                                           # output sound size
	yhw = np.zeros(N)                                        # initialize output sound frame
	ysw = np.zeros(N)                                        # initialize output sound frame
	yh = np.zeros(ysize)                                      # initialize output array
	yst = np.zeros(ysize)                                     # initialize output array
	sw = np.zeros(N)     
	ow = triang(2*H)                                          # overlapping window
	sw[hN-H:hN+H] = ow      
	bh = blackmanharris(N)                                   # synthesis window
	bh = bh / sum(bh)                                         # normalize synthesis window
	wr = bh                                                   # window for residual
	sw[hN-H:hN+H] = sw[hN-H:hN+H] / bh[hN-H:hN+H]             # synthesis window for harmonic component
	sws = H*hanning(N)/2                                      # synthesis window for stochastic component
	lastyhfreq = hfreq[0,:]                                   # initialize synthesis harmonic frequencies
	yhphase = 2*np.pi*np.random.rand(nH)                      # initialize synthesis harmonic phases     
	for l in range(L):
		yhfreq = hfreq[l,:]                                     # synthesis harmonics frequencies
		yhmag = hmag[l,:]                                       # synthesis harmonic amplitudes
		mYrenv = mYst[l,:]                                      # synthesis residual envelope
		if (hphase.size > 0):
			yhphase = hphase[l,:] 
		else:
			yhphase += (np.pi*(lastyhfreq+yhfreq)/fs)*H             # propagate phases
		lastyhfreq = yhfreq
		Yh = UF.genSpecSines(yhfreq, yhmag, yhphase, N, fs)     # generate spec sines 
		mYs = resample(mYrenv, hN)                              # interpolate to original size
		mYs = 10**(mYs/20)                                      # dB to linear magnitude  
		pYs = 2*np.pi*np.random.rand(hN)                        # generate phase random values
		Ys = np.zeros(N, dtype = complex)
		Ys[:hN] = mYs * np.exp(1j*pYs)                         # generate positive freq.
		Ys[hN+1:] = mYs[:0:-1] * np.exp(-1j*pYs[:0:-1])        # generate negative freq.
		fftbuffer = np.zeros(N)
		fftbuffer = np.real(ifft(Yh))                           # inverse FFT of harm spectrum
		yhw[:hN-1] = fftbuffer[hN+1:]                         # undo zer-phase window
		yhw[hN-1:] = fftbuffer[:hN+1] 
		fftbuffer = np.zeros(N)
		fftbuffer = np.real(ifft(Ys))                           # inverse FFT of stochastic approximation spectrum
		ysw[:hN-1] = fftbuffer[hN+1:]                           # undo zero-phase window
		ysw[hN-1:] = fftbuffer[:hN+1]
		yh[pout:pout+N] += sw*yhw                               # overlap-add for sines
		yst[pout:pout+N] += sws*ysw                             # overlap-add for stoch
		pout += H                                               # advance sound pointer
	y = yh+yst                                                # sum harmonic and stochastic components
	return y, yh, yst


def hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, stocf):
	# Analysis/synthesis of a sound using the harmonic plus stochastic model
	# x: input sound, fs: sampling rate, w: analysis window, 
	# N: FFT size (minimum 512), t: threshold in negative dB, 
	# nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
	# maxf0: maximim f0 frequency in Hz, 
	# f0et: error threshold in the f0 detection (ex: 5),
	# stocf: decimation factor of mag spectrum for stochastic analysis
	# returns y: output sound, yh: harmonic component, yst: stochastic component

	hN = N/2                                               # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                    # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                        # half analysis window size by floor
	Ns = 512                                               # FFT size for synthesis (even)
	H = Ns/4                                               # Hop size used for analysis and synthesis
	hNs = Ns/2      
	pin = max(hNs, hM1)                                    # initialize sound pointer in middle of analysis window          
	pend = x.size - max(hNs, hM1)                          # last sample to start a frame
	fftbuffer = np.zeros(N)                                # initialize buffer for FFT
	yhw = np.zeros(Ns)                                     # initialize output sound frame
	ystw = np.zeros(Ns)                                    # initialize output sound frame
	yh = np.zeros(x.size)                                  # initialize output array
	yst = np.zeros(x.size)                                 # initialize output array
	w = w / sum(w)                                         # normalize analysis window
	sw = np.zeros(Ns)     
	ow = triang(2*H)                                       # overlapping window
	sw[hNs-H:hNs+H] = ow      
	bh = blackmanharris(Ns)                                # synthesis window
	bh = bh / sum(bh)                                      # normalize synthesis window
	wr = bh                                                # window for residual
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]    # synthesis window for harmonic component
	sws = H*hanning(Ns)/2                                  # synthesis window for stochastic
	hfreqp = []
	f0t = 0
	f0stable = 0
	while pin<pend:  
	#-----analysis-----             
		x1 = x[pin-hM1:pin+hM2]                              # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                   # compute dft
		ploc = UF.peakDetection(mX, hN, t)                   # find peaks                
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values
		ipfreq = fs * iploc/N                                # convert peak locations to Hz
		f0t = UF.f0DetectionTwm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
		if ((f0stable==0)&(f0t>0)) \
			or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
			f0stable = f0t                                # consider a stable f0 if it is close to the previous one
		else:
			f0stable = 0
		hfreq, hmag, hphase = UF.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs) # find harmonics
		hfreqp = hfreq
		ri = pin-hNs-1                                       # input sound pointer for residual analysis
		xw2 = x[ri:ri+Ns]*wr                                 # window the input sound                                       
		fftbuffer = np.zeros(Ns)                             # reset buffer
		fftbuffer[:hNs] = xw2[hNs:]                          # zero-phase window in fftbuffer
		fftbuffer[hNs:] = xw2[:hNs]                           
		X2 = fft(fftbuffer)                                  # compute FFT for residual analysis
	#-----synthesis-----
		Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)    # generate spec sines of harmonic component          
		Xr = X2-Yh;                                          # get the residual complex spectrum
		mXr = 20 * np.log10(abs(Xr[:hNs]))                   # magnitude spectrum of residual
		mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf) # decimate the magnitude spectrum and avoid -Inf                     
		mYst = resample(mXrenv, hNs)                         # interpolate to original size
		mYst = 10**(mYst/20)                                 # dB to linear magnitude  
		pYst = 2*np.pi*np.random.rand(hNs)                   # generate phase random values
		Yst = np.zeros(Ns, dtype = complex)
		Yst[:hNs] = mYst * np.exp(1j*pYst)                   # generate positive freq.
		Yst[hNs+1:] = mYst[:0:-1] * np.exp(-1j*pYst[:0:-1])  # generate negative freq.
		
		fftbuffer = np.zeros(Ns)
		fftbuffer = np.real(ifft(Yh))                         # inverse FFT of harmonic spectrum
		yhw[:hNs-1] = fftbuffer[hNs+1:]                       # undo zero-phase window
		yhw[hNs-1:] = fftbuffer[:hNs+1] 

		fftbuffer = np.zeros(Ns)
		fftbuffer = np.real(ifft(Yst))                        # inverse FFT of stochastic spectrum
		ystw[:hNs-1] = fftbuffer[hNs+1:]                      # undo zero-phase window
		ystw[hNs-1:] = fftbuffer[:hNs+1]

		yh[ri:ri+Ns] += sw*yhw                                # overlap-add for sines
		yst[ri:ri+Ns] += sws*ystw                             # overlap-add for stochastic
		pin += H                                              # advance sound pointer
	
	y = yh+yst                                              # sum of harmonic and stochastic components
	return y, yh, yst

if __name__ == '__main__':
	(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase.wav'))
	w = np.blackman(601)
	N = 1024
	t = -100
	nH = 100
	minf0 = 350
	maxf0 = 700
	f0et = 5
	maxnpeaksTwm = 5
	minSineDur = .1
	harmDevSlope = 0.01
	Ns = 512
	H = Ns/4
	stocf = .2
	hfreq, hmag, hphase, mYst = hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
	y, yh, yst = hpsModelSynth(hfreq, hmag, hphase, mYst, Ns, H, fs)
 
	UF.play(y, fs)
	UF.play(yh, fs)
	UF.play(yst, fs)

	plt.figure(1, figsize=(9.5, 7)) 
	maxplotfreq = 20000.0
	numFrames = int(mYst[:,0].size)
	sizeEnv = int(mYst[0,:].size)
	frmTime = H*np.arange(numFrames)/float(fs)
	binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
	plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:sizeEnv*maxplotfreq/(.5*fs)+1]))
	plt.autoscale(tight=True)

	harms = hfreq*np.less(hfreq,maxplotfreq)
	harms[harms==0] = np.nan
	numFrames = int(harms[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs) 
	plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
	plt.autoscale(tight=True)
	plt.title('harmonic + stochastic components')

	plt.tight_layout()
	plt.show()
