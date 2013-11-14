import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../code/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../code/basicFunctions_C/'))

import smsF0DetectionTwm as fd
import smsWavplayer as wp
import smsPeakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


def hprModelPlot(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxFreq,maxnpeaks=10): 
	hN = N/2                                                      # size of positive spectrum
	hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
	hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
	Ns = 512                                                      # FFT size for synthesis (even)
	H = Ns/4                                                      # Hop size used for analysis and synthesis
	hNs = Ns/2      
	pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
	pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
	fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
	yhw = np.zeros(Ns)                                            # initialize output sound frame
	yrw = np.zeros(Ns)                                            # initialize output sound frame
	yh = np.zeros(x.size)                                         # initialize output array
	yr = np.zeros(x.size)                                         # initialize output array
	w = w / sum(w)                                                # normalize analysis window
	sw = np.zeros(Ns)     
	ow = triang(2*H)                                              # overlapping window
	sw[hNs-H:hNs+H] = ow      
	bh = blackmanharris(Ns)                                       # synthesis window
	bh = bh / sum(bh)                                             # normalize synthesis window
	wr = bh                                                       # window for residual
	sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]

	numFrames = int(math.floor(pend/float(H)))
	frmNum = 0
	frmTime = []
	lastBin = N*maxFreq/float(fs)
	binFreq = np.arange(lastBin)*float(fs)/N       # The bin frequencies
    
	while pin<pend:                                         # while sound pointer is smaller than last sample    
		frmTime.append(pin/float(fs))         
		xw = x[pin-hM1:pin+hM2] * w                                  # window the input sound
		fftbuffer = np.zeros(N)                                      # reset buffer
		fftbuffer[:hM1] = xw[hM2:]                                   # zero-phase window in fftbuffer
		fftbuffer[N-hM2:] = xw[:hM2]                           
		X = fft(fftbuffer)                                           # compute FFT
		mX = 20 * np.log10(abs(X[:hN]))                              # magnitude spectrum of positive frequencies
		ploc = PP.peakDetection(mX, hN, t)                
		pX = np.unwrap(np.angle(X[:hN]))                             # unwrapped phase spect. of positive freq.    
		iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)          # refine peak values
    
		f0 = fd.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaks)  # find f0
		hloc = np.zeros(nH)                                          # initialize harmonic locations
		hmag = np.zeros(nH)-100                                      # initialize harmonic magnitudes
		hphase = np.zeros(nH)                                        # initialize harmonic phases
		hf = (f0>0)*(f0*np.arange(1, nH+1))                          # initialize harmonic frequencies
		hi = 0                                                       # initialize harmonic index
		npeaks = ploc.size;                                          # number of peaks found
    
		while f0>0 and hi<nH and hf[hi]<fs/2 :                       # find harmonic peaks
			dev = min(abs(iploc/N*fs - hf[hi]))
			pei = np.argmin(abs(iploc/N*fs - hf[hi]))                  # closest peak
			if ( hi==0 or not any(hloc[:hi]==iploc[pei]) ) and dev<maxhd*hf[hi] :
				hloc[hi] = iploc[pei]                                    # harmonic locations
				hmag[hi] = ipmag[pei]                                    # harmonic magnitudes
				hphase[hi] = ipphase[pei]                                # harmonic phases
			hi += 1                                                    # increase harmonic index
 
		if frmNum == 0:                                       # Accumulate and store STFT
			XSpec = np.transpose(np.array([mX[:lastBin]]))
			ind1 = np.where(hloc>0)[0]
			ind2 = np.where(hloc<=lastBin)[0]
			ind = list((set(ind1.tolist())&set(ind2.tolist())))
			final_peaks = hloc[ind]
			parray = np.zeros([final_peaks.size,2])
			parray[:,0]=pin/float(fs)
			parray[:,1]=final_peaks*float(fs)/N
			specPeaks = parray
		else:
			XSpec = np.hstack((XSpec,np.transpose(np.array([mX[:lastBin]]))))
			ind1 = np.where(hloc>0)[0]
			ind2 = np.where(hloc<=lastBin)[0]
			ind = list((set(ind1.tolist())&set(ind2.tolist())))
			final_peaks = hloc[ind]
			parray = np.zeros([final_peaks.size,2])
			parray[:,0]=pin/float(fs)
			parray[:,1]=final_peaks*float(fs)/N
			specPeaks = np.append(specPeaks, parray,axis=0)
		
		hloc[:hi] = (hloc[:hi]!=0) * (hloc[:hi]*Ns/N)                # synth. locs
		ri = pin-hNs-1                                               # input sound pointer for residual analysis
		xr = x[ri:ri+Ns]*wr                                          # window the input sound                                       
		fftbuffer = np.zeros(Ns)                                     # reset buffer
		fftbuffer[:hNs] = xr[hNs:]                                   # zero-phase window in fftbuffer
		fftbuffer[hNs:] = xr[:hNs]                           
		Xr = fft(fftbuffer)                                          # compute FFT for residual analysis
		Yh = GS.genSpecSines(hloc[:hi], hmag, hphase, Ns)            # generate spec sines of harmonic component          
		Yr = Xr-Yh;                                                  # get the residual complex spectrum
		mYr = 20 * np.log10(abs(Yr[:hNs]))
		lastBinYr = Ns*maxFreq/float(fs)
		binFreqYr = np.arange(lastBinYr)*float(fs)/Ns       # The bin frequencies
		if frmNum == 0:                                        # Accumulate and store STFT
			YrSpec = np.transpose(np.array([mYr[:lastBinYr]]))
		else:
			YrSpec = np.hstack((YrSpec,np.transpose(np.array([mYr[:lastBinYr]]))))
		pin += H
		frmNum += 1
	
	frmTime = np.array(frmTime)                               # The time at the centre of the frames
	plt.figure(1)
	plt.subplot(2,1,1)
	plt.hold(True)
	plt.pcolormesh(frmTime,binFreq,XSpec)
	plt.scatter(specPeaks[:,0]+(0.5*H/float(fs)), specPeaks[:,1], s=10, marker='x')
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
	plt.autoscale(tight=True)
	plt.title('X spectrogram + peaks')

	plt.subplot(2,1,2)
	plt.hold(True)
	plt.pcolormesh(frmTime,binFreqYr,YrSpec)
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
	plt.autoscale(tight=True)
	plt.title('residual spectrogram')

	plt.show()

# example call of hprModelPlot function
if __name__ == '__main__':
    (fs, x) = wp.wavread('../../sounds/sax-phrase-short.wav')
    w = np.blackman(901)
    N = 2048
    t = -70
    nH = 10
    minf0 = 300
    maxf0 = 650
    f0et = 5
    maxhd = 0.2
    maxFreq = 3000.0
    maxnpeaks=10
    hprModelPlot(x,fs,w,N,t,nH, minf0, maxf0, f0et, maxhd, maxFreq, maxnpeaks)
   