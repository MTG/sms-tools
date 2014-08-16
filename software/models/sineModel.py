# functions that implement analysis and synthesis of sounds using the Sinusoidal Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift
import math
import dftModel as DFT
import utilFunctions as UF
	
def sineModel(x, fs, w, N, t):
	# Analysis/synthesis of a sound using the sinusoidal model, without sine tracking
	# x: input array sound, w: analysis window, N: size of complex spectrum, t: threshold in negative dB 
	# returns y: output array sound
	hN = N/2                                                # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	Ns = 512                                                # FFT size for synthesis (even)
	H = Ns/4                                                # Hop size used for analysis and synthesis
	hNs = Ns/2                                              # half of synthesis FFT size
	pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
	pend = x.size - max(hNs, hM1)                           # last sample to start a frame
	fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
	yw = np.zeros(Ns)                                       # initialize output sound frame
	y = np.zeros(x.size)                                    # initialize output array
	w = w / sum(w)                                          # normalize analysis window
	sw = np.zeros(Ns)                                       # initialize synthesis window
	ow = triang(2*H)                                        # triangular window
	sw[hNs-H:hNs+H] = ow                                    # add triangular window
	bh = blackmanharris(Ns)                                 # blackmanharris window
	bh = bh / sum(bh)                                       # normalized blackmanharris window
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
	while pin<pend:                                         # while input sound pointer is within sound 
	#-----analysis-----             
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
		ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
		pmag = mX[ploc]                                       # get the magnitude of the peaks
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
	#-----synthesis-----
		plocs = iploc*Ns/N                                    # adapt peak locations to size of synthesis FFT
		Y = UF.genSpecSines(fs*plocs/N, ipmag, ipphase, Ns, fs)    # generate sines in the spectrum         
		fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
		yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
		yw[hNs-1:] = fftbuffer[:hNs+1] 
		y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
		pin += H                                              # advance sound pointer
	return y

def sineModelAnal(x, fs, w, N, H, t, maxnSines = 100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
	# Analysis of a sound using the sinusoidal model with sine tracking
	# x: input array sound, w: analysis window, N: size of complex spectrum, H: hop-size, t: threshold in negative dB
	# maxnSines: maximum number of sines per frame, minSineDur: minimum duration of sines in seconds
	# freqDevOffset: minimum frequency deviation at 0Hz, freqDevSlope: slope increase of minimum frequency deviation
	# returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoidal tracks
	hN = N/2                                                # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
	x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
	x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
	pin = hM1                                               # initialize sound pointer in middle of analysis window       
	pend = x.size - hM1                                     # last sample to start a frame
	w = w / sum(w)                                          # normalize analysis window
	tfreq = np.array([])
	while pin<pend:                                         # while input sound pointer is within sound            
		x1 = x[pin-hM1:pin+hM2]                               # select frame
		mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
		ploc = UF.peakDetection(mX, t)                        # detect locations of peaks
		pmag = mX[ploc]                                       # get the magnitude of the peaks
		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
		ipfreq = fs*iploc/float(N)                            # convert peak locations to Hertz
		# perform sinusoidal tracking by adding peaks to trajectories
		tfreq, tmag, tphase = UF.sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
		tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))  # limit number of tracks to maxnSines
		tmag = np.resize(tmag, min(maxnSines, tmag.size))     # limit number of tracks to maxnSines
		tphase = np.resize(tphase, min(maxnSines, tphase.size)) # limit number of tracks to maxnSines
		jtfreq = np.zeros(maxnSines)                          # temporary output array
		jtmag = np.zeros(maxnSines)                           # temporary output array
		jtphase = np.zeros(maxnSines)                         # temporary output array   
		jtfreq[:tfreq.size]=tfreq                             # save track frequencies to temporary array
		jtmag[:tmag.size]=tmag                                # save track magnitudes to temporary array
		jtphase[:tphase.size]=tphase                          # save track magnitudes to temporary array
		if pin == hM1:                                        # if first frame initialize output sine tracks
			xtfreq = jtfreq 
			xtmag = jtmag
			xtphase = jtphase
		else:                                                 # rest of frames append values to sine tracks
			xtfreq = np.vstack((xtfreq, jtfreq))
			xtmag = np.vstack((xtmag, jtmag))
			xtphase = np.vstack((xtphase, jtphase))
		pin += H
	# delete sine tracks shorter than minSineDur
	xtfreq = UF.cleaningSineTracks(xtfreq, round(fs*minSineDur/H))  
	return xtfreq, xtmag, xtphase


def sinewaveSynth(freqs, amp, H, fs):
	# Synthesis of one sinusoid with time-varying frequency
	# freqs, amps: array of frequencies and amplitudes of sinusoids
	# H: hop size, fs: sampling rate
	# returns y: output array sound
	t = np.arange(H)/float(fs)                              # time array
	lastphase = 0                                           # initialize synthesis phase
	lastfreq = freqs[0]                                     # initialize synthesis frequency
	y = np.array([])                                        # initialize output array
	for l in range(freqs.size):                             # iterate over all frames
		if (lastfreq==0) & (freqs[l]==0):                     # if 0 freq add zeros
			A = np.zeros(H)
			freq = np.zeros(H)
		elif (lastfreq==0) & (freqs[l]>0):                    # if starting freq ramp up the amplitude
			A = np.arange(0,amp, amp/H)
			freq = np.ones(H)*freqs[l]
		elif (lastfreq>0) & (freqs[l]>0):                     # if freqs in boundaries use both
			A = np.ones(H)*amp
			if (lastfreq==freqs[l]):
				freq = np.ones(H)*lastfreq
			else:
				freq = np.arange(lastfreq,freqs[l], (freqs[l]-lastfreq)/H)
		elif (lastfreq>0) & (freqs[l]==0):                    # if ending freq ramp down the amplitude
			A = np.arange(amp,0,-amp/H)
			freq = np.ones(H)*lastfreq
		phase = 2*np.pi*freq*t+lastphase                      # generate phase values
		yh = A * np.cos(phase)                                # compute sine for one frame
		lastfreq = freqs[l]                                   # save frequency for phase propagation
		lastphase = np.remainder(phase[H-1], 2*np.pi)         # save phase to be use for next frame
		y = np.append(y, yh)                                  # append frame to previous one
	return y

def sineModelSynth(tfreq, tmag, tphase, N, H, fs):
	# Synthesis of a sound using the sinusoidal model
	# tfreq,tmag,tphase: frequencies, magnitudes and phases of sinusoids
	# N: synthesis FFT size, H: hop size, fs: sampling rate
	# returns y: output array sound
	hN = N/2                                                # half of FFT size for synthesis
	L = tfreq[:,0].size                                     # number of frames
	pout = 0                                                # initialize output sound pointer         
	ysize = H*(L+3)                                         # output sound size
	y = np.zeros(ysize)                                     # initialize output array
	sw = np.zeros(N)                                        # initialize synthesis window
	ow = triang(2*H)                                        # triangular window
	sw[hN-H:hN+H] = ow                                      # add triangular window
	bh = blackmanharris(N)                                  # blackmanharris window
	bh = bh / sum(bh)                                       # normalized blackmanharris window
	sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
	lastytfreq = tfreq[0,:]                                 # initialize synthesis frequencies
	ytphase = 2*np.pi*np.random.rand(tfreq[0,:].size)       # initialize synthesis phases 
	for l in range(L):                                      # iterate over all frames
		if (tphase.size > 0):                                 # if no phases generate them
			ytphase = tphase[l,:] 
		else:
			ytphase += (np.pi*(lastytfreq+tfreq[l,:])/fs)*H     # propagate phases
		Y = UF.genSpecSines(tfreq[l,:], tmag[l,:], ytphase, N, fs)  # generate sines in the spectrum         
		lastytfreq = tfreq[l,:]                               # save frequency for phase propagation
		yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
		y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
		pout += H                                             # advance sound pointer
	y = np.delete(y, range(hN))                             # delete half of first window
	y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window 
	return y
