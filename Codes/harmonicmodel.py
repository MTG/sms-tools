import numpy as np
import UtilityFunctions as uf
import matplotlib.pyplot as plt
import wavplayer as wp
from scipy.io.wavfile import read
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import f0detectiontwm as fd

ef genspecsines(iploc, ipmag, ipphase, N):
  # Generate a spectrum from a series of sine values
  # iploc, ipmag, ipphase: sine peaks locations, magnitudes and phases
  # N: size of complex spectrum
  # returns Y: generated complex spectrum of sines

  Y = np.zeros(N, dtype = complex)                 # initialize output spectrum  
  hN = N/2                                         # size of positive freq. spectrum

  for i in range(0, iploc.size):                   # generate all sine spectral lobes
    loc = iploc[i]                                 # it should be in range ]0,hN-1[

    if loc<1 or loc>hN-1: continue
    binremainder = round(loc)-loc;
    lb = np.arange(binremainder-4, binremainder+5) # main lobe (real value) bins to read
    lmag = uf.genbh92lobe(lb) * 10**(ipmag[i]/20)  # lobe magnitudes of the complex exponential
    b = np.arange(round(loc)-4, round(loc)+5)
    
    for m in range(0, 9):
      if b[m] < 0:                                 # peak lobe crosses DC bin
        Y[-b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
      elif b[m] > hN:                              # peak lobe croses Nyquist bin
        Y[b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
      elif b[m] == 0 or b[m] == hN:                # peak lobe in the limits of the spectrum 
        Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i]) + lmag[m]*np.exp(-1j*ipphase[i])
      else:                                        # peak lobe in positive freq. range
        Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i])
    
    Y[hN+1:] = Y[hN-1:0:-1].conjugate()            # fill the negative part of the spectrum
  
  return Y

def peak_interp(mX, pX, ploc):
  # interpolate peak values using parabolic interpolation
  # mX: magnitude spectrum, pX: phase spectrum, ploc: locations of peaks
  # returns iploc, ipmag, ipphase: interpolated values
  
  val = mX[ploc]                                          # magnitude of peak bin 
  lval = mX[ploc-1]                                       # magnitude of bin at left
  rval = mX[ploc+1]                                       # magnitude of bin at right
  iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
  ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
  ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks

  return iploc, ipmag, ipphase

def peak_detection(mX, hN, t):
  # detect spectral peak locations
  # mX: magnitude spectrum, hN: half number of samples, t: threshold
  # returns ploc: peak locations

  thresh = np.where(mX[1:hN-1]>t, mX[1:hN-1], 0);          # locations above threshold
  next_minor = np.where(mX[1:hN-1]>mX[2:], mX[1:hN-1], 0)  # locations higher than the next one
  prev_minor = np.where(mX[1:hN-1]>mX[:hN-2], mX[1:hN-1], 0) # locations higher than the previous one
  ploc = thresh * next_minor * prev_minor                  # locations fulfilling the three criteria
  ploc = ploc.nonzero()[0] + 1

  return ploc

def harmonic_model(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd):
  # Analysis/synthesis of a sound using the sinusoidal harmonic model
  # x: input sound, fs: sampling rate, w: analysis window (odd size), 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # yh: harmonic component, yr: residual component
  # returns y: output sound

  hN = N/2                                                      # size of positive spectrum
  hM = (w.size+1)/2                                             # half analysis window size
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM)                                            # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM)                                  # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  yh = np.zeros(Ns)                                             # initialize output sound frame
  y = np.zeros(x.size)                                          # initialize output array
  x = np.float32(x) / (2**15)                                   # normalize input signal
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)                                             # initialize synthesis window
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]

  while pin<pend:       
            
  #-----analysis-----             
    xw = x[pin-hM:pin+hM-1] * w                                  # window the input sound
    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM] = xw[hM-1:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]                           
    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
    ploc = peak_detection(mX, hN, t)                             # detect peak locations
    pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.     
    iploc, ipmag, ipphase = peak_interp(mX, pX, ploc)            # refine peak values
    
    f0 = fd.f0detectiontwm(iploc, ipmag, N, fs, f0et, minf0, maxf0)  # find f0
    hloc = np.zeros(nH)                                          # initialize harmonic locations
    hmag = np.zeros(nH)-100                                      # initialize harmonic magnitudes
    hphase = np.zeros(nH)                                        # initialize harmonic phases
    hf = (f0>0)*(f0*np.arange(1, nH+1))                          # initialize harmonic frequencies
    hi = 0                                                       # initialize harmonic index
    npeaks = ploc.size                                           # number of peaks found
    
    while f0>0 and hi<nH and hf[hi]<fs/2 :                       # find harmonic peaks
      dev = min(abs(iploc/N*fs - hf[hi]))
      pei = np.argmin(abs(iploc/N*fs - hf[hi]))                  # closest peak
      if ( hi==0 or not any(hloc[:hi]==iploc[pei]) ) and dev<maxhd*hf[hi] :
        hloc[hi] = iploc[pei]                                    # harmonic locations
        hmag[hi] = ipmag[pei]                                    # harmonic magnitudes
        hphase[hi] = ipphase[pei]                                # harmonic phases
      hi += 1                                                    # increase harmonic index
    
    hloc = (hloc!=0) * (hloc*Ns/N)                               # synth. locs

  #-----synthesis-----
    Yh = genspecsines(hloc, hmag, hphase, Ns)                    # generate spec sines          
    fftbuffer = np.real( ifft(Yh) )                              # inverse FFT
    yh[:hNs-1] = fftbuffer[hNs+1:]                               # undo zero-phase window
    yh[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yh                                  # overlap-add
    pin += H                                                     # advance sound pointer
  
  return y


(fs, x) = read('speech-female.wav')
wp.play(x, fs)

w = np.hamming(1025)
N = 1024
t = -120
nH = 30
minf0 = 200
maxf0 = 500
f0et = 2
maxhd = 0.2
y = harmonic_model(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd)

y *= 2**15
y = y.astype(np.int16)

wp.play(y, fs)