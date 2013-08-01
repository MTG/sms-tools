import numpy as np
import UtilityFunctions as uf
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import wavplayer as wp
from scipy.io.wavfile import read
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import time
import f0detectiontwm as fd
from pylab import specgram

def genspecsines(iploc, ipmag, ipphase, N):
  # Compute a spectrum from a series of sine values
  # iploc, ipmag, ipphase: sine locations, magnitudes and phases
  # N: size of complex spectrum
  # Y: generated complex spectrum of sines

  Y = np.zeros(N, dtype = complex)                 # initialize output spectrum  
  hN = N/2                                         # size of positive freq. spectrum

  for i in range(0, iploc.size):                   # generate all sine spectral lobes
    loc = iploc[i]                                 # it should be in range ]0,hN-1[

    if loc<1 or loc>hN-1: continue
    binremainder = round(loc)-loc
    lb = np.arange(binremainder-4, binremainder+5) # main lobe (real value) bins to read
    lmag = uf.genbh92lobe(lb) * 10**(ipmag[i]/20)     # lobe magnitudes of the complex exponential
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
    
    Y[hN+1:] = Y[hN-1:0:-1].conjugate()            # fill the rest of the spectrum
  
  return Y

def peak_interp(mX, pX, ploc):
  # mX: magnitude spectrum, pX: phase spectrum, ploc: locations of peaks
  # iploc, ipmag, ipphase: interpolated values
  
  val = mX[ploc]                                          # magnitude of peak bin 
  lval = mX[ploc-1]                                       # magnitude of bin at left
  rval = mX[ploc+1]                                       # magnitude of bin at right
  iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
  ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
  ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks

  return iploc, ipmag, ipphase

def peak_detection(mX, hN, t):
  # mX: magnitude spectrum, hN: half number of samples, t: threshold
  # to be a peak it has to accomplish three conditions:

  thresh = np.where(mX[1:hN-1]>t, mX[1:hN-1], 0)
  next_minor = np.where(mX[1:hN-1]>mX[2:], mX[1:hN-1], 0)
  prev_minor = np.where(mX[1:hN-1]>mX[:hN-2], mX[1:hN-1], 0)
  ploc = thresh * next_minor * prev_minor
  ploc = ploc.nonzero()[0] + 1

  return ploc

def hps(x, fs, w=np.hamming(801), N=1024, t=-120, nH=30, minf0=200, maxf0=500, f0et=5, maxhd=0.2, stocf=0.5):
  # Analysis/synthesis of a sound using the harmonic plus stochastic model
  # x: input sound, fs: sampling rate, w: analysis window (odd size), 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # y: output sound, yh: harmonic component, ys: stochastic component

  x = np.float32(x) / (2**15)                                   # normalize input signal

  hN = N/2                                                      # size of positive spectrum
  hM = (w.size+1)/2                                             # half analysis window size
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM)                                            # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM)                                  # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  yhw = np.zeros(Ns)                                            # initialize output sound frame
  ysw = np.zeros(Ns)                                            # initialize output sound frame
  yh = np.zeros(x.size)                                         # initialize output array
  ys = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  sws = H*hanning(Ns)/2                                         # synthesis window for stochastic
  lastyhloc = np.zeros(nH)                                      # initialize synthesis harmonic locations
  yhphase = 2*np.pi * np.random.rand(nH)                        # initialize synthesis harmonic phases     

  while pin<pend:

  #-----analysis-----             

    xw = x[pin-hM:pin+hM-1] * w                                  # window the input sound

    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM] = xw[hM-1:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]                           

    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
    ploc = peak_detection(mX, hN, t)                
    pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.     
    iploc, ipmag, ipphase = peak_interp(mX, pX, ploc)            # refine peak values 

    f0 = fd.f0detectiontwm(iploc, ipmag, N, fs, f0et, minf0, maxf0)  # find f0

    hloc = np.zeros(nH)                                          # initialize harmonic locations
    hmag = np.zeros(nH)-100                                      # initialize harmonic magnitudes
    hphase = np.zeros(nH)                                        # initialize harmonic phases
    hf = (f0>0) * (f0*np.arange(1, nH+1))                        # initialize harmonic frequencies
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
    
    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                         # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                  # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]                            
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
    
    Xh = genspecsines(hloc, hmag, hphase, Ns)                    # generate sines
    Xr = X2-Xh                                                   # get the residual complex spectrum
    mXr = 20 * np.log10( abs(Xr[:hNs]) )                         # magnitude spectrum of residual
    mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)     # decimate the magnitude spectrum and avoid -Inf

  #-----synthesis data-----
    yhloc = hloc                                                 # synthesis harmonics locs
    yhmag = hmag                                                 # synthesis harmonic amplitudes
    mYrenv = mXrenv                                              # synthesis residual envelope
    yf0 = f0  

  #-----transformations-----

  #-----synthesis-----
    yhphase += 2*np.pi * (lastyhloc+yhloc)/2/Ns*H                # propagate phases
    lastyhloc = yhloc 
    
    Yh = genspecsines(yhloc, yhmag, yhphase, Ns)                 # generate spec sines 
    mYs = resample(mYrenv, hNs)                                  # interpolate to original size
    mYs = 10**(mYs/20)                                           # dB to linear magnitude  
    if f0>0:
        mYs *= np.cos(np.pi*np.arange(0, hNs)/Ns*fs/yf0)**2      # filter residual

    fc = 1+round(500.0/fs*Ns)                                    # 500 Hz
    mYs[:fc] *= (np.arange(0, fc)/(fc-1))**2                     # HPF
    pYs = 2*np.pi * np.random.rand(hNs)                          # generate phase random values
    
    Ys = np.zeros(Ns, dtype = complex)
    Ys[:hNs] = mYs * np.exp(1j*pYs)                              # generate positive freq.
    Ys[hNs+1:] = mYs[:0:-1] * np.exp(-1j*pYs[:0:-1])             # generate negative freq.

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Yh) )                            

    yhw[:hNs-1] = fftbuffer[hNs+1:]                              # sines in time domain using IFFT
    yhw[hNs-1:] = fftbuffer[:hNs+1] 

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Ys) )
    ysw[:hNs-1] = fftbuffer[hNs+1:]                              # stochastic in time domain using IFFT
    ysw[hNs-1:] = fftbuffer[:hNs+1]

    yh[ri:ri+Ns] += sw*yhw                                       # overlap-add for sines
    ys[ri:ri+Ns] += sws*ysw                                      # overlap-add for stoch
    pin += H                                                     # advance sound pointer

  y = yh+ys
  return y, yh, ys


(fs, x) = read('speech-female.wav')
# wp.play(x, fs)

w = np.hamming(1025)
N = 1024
t = -120
nH = 30
minf0 = 200
maxf0 = 500
f0et = 5
maxhd = 0.2
stocf = 0.5
y, yh, ys = hps(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf)

y *= 2**15
y = y.astype(np.int16)

yh *= 2**15
yh = yh.astype(np.int16)

ys *= 2**15
ys = ys.astype(np.int16)

wp.play(y, fs)
wp.play(yh, fs)
wp.play(ys, fs)