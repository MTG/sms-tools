import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
import sineModel as SM
import stft as STFT
import utilFunctions as UF
  
  
def sprModel(x, fs, w, N, t):
  # Analysis/synthesis of a sound using the sinusoidal plus residual model, one frame at a time
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # returns y: output sound, ys: sinusoidal component, xr: residual component

  hN = N/2                                                      # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  ysw = np.zeros(Ns)                                            # initialize output sound frame
  xrw = np.zeros(Ns)                                            # initialize output sound frame
  ys = np.zeros(x.size)                                         # initialize output array
  xr = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  while pin<pend:  
  #-----analysis-----             
    xw = x[pin-hM1:pin+hM2] * w                                  # window the input sound
    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM1] = xw[hM2:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM2:] = xw[:hM2]                           
    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10(abs(X[:hN]))                              # magnitude spectrum of positive frequencies
    ploc = UF.peakDetection(mX, hN, t)                
    pX = np.unwrap(np.angle(X[:hN]))                             # unwrapped phase spect. of positive freq.    
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)          # refine peak values
    iploc = (iploc!=0) * (iploc*Ns/N)                            # synth. locs
    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                          # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                   # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]                           
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
  #-----synthesis-----
    Ys = UF.genSpecSines(fs*iploc/N, ipmag, ipphase, Ns, fs)     # generate spec of sinusoidal component          
    Xr = X2-Ys;                                                  # get the residual complex spectrum
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Ys))                                # inverse FFT of sinusoidal spectrum
    ysw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    ysw[hNs-1:] = fftbuffer[:hNs+1] 
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Xr))                                # inverse FFT of residual spectrum
    xrw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    xrw[hNs-1:] = fftbuffer[:hNs+1]
    ys[ri:ri+Ns] += sw*ysw                                       # overlap-add for sines
    xr[ri:ri+Ns] += sw*xrw                                       # overlap-add for residual
    pin += H                                                     # advance sound pointer
  y = ys+xr                                                      # sum of sinusoidal and residual components
  return y, ys, xr


# example of using the sinusoidal plus residual model by first doing the whole analysis and then the synthesis
# this permits to apply sinusoidal tracking, which is not possible with the sprModel function
if __name__ == '__main__':

  # read bendir sound
	(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
	w = np.hamming(2001)          # compute analysis window of odd size
	N = 2048                      # fft size
	H = 128                       # hop size of analysis window
	t = -100                      # magnitude threshold used for peak detection
	minSineDur = .02              # minimum length of sinusoidal tracks in seconds
	maxnSines = 200               # maximum number of paralell sinusoids
	freqDevOffset = 10            # allowed deviation in Hz at lowest frequency used in the frame to frame tracking
	freqDevSlope = 0.001          # increase factor of the deviation as the frequency increases

	# perform sinusoidal analysis
	tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
	
  # subtract sinusoids from original 
	xr = UF.sineSubtraction(x, N, H, tfreq, tmag, tphase, fs)
  
  # compute spectrogram of residual
	mXr, pXr = STFT.stftAnal(xr, fs, hamming(H*2), H*2, H)
	Ns = 512
	
  # synthesize sinusoids
	ys = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

	# plot magnitude spectrogram of residual
	plt.figure(1, figsize=(9.5, 7))
	numFrames = int(mXr[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs)                             
	binFreq = np.arange(H)*float(fs)/(H*2)                       
	plt.pcolormesh(frmTime, binFreq, np.transpose(mXr))
	plt.autoscale(tight=True)

	# plot sinusoidal frequencies on top of residual spectrogram
	tfreq[tfreq==0] = np.nan
	numFrames = int(tfreq[:,0].size)
	frmTime = H*np.arange(numFrames)/float(fs) 
	plt.plot(frmTime, tfreq, color='k', ms=3, alpha=1)
	plt.xlabel('Time(s)')
	plt.ylabel('Frequency(Hz)')
	plt.autoscale(tight=True)
	plt.title('sinusoidal + residual components')

	# write sounds files for sinusoidal sound and residual sound
	UF.wavwrite(ys, fs, 'bendir-sines.wav')
	UF.wavwrite(xr, fs, 'bendir-residual.wav')

	plt.tight_layout()
	plt.show()
