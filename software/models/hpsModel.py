import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal, dftSynth
import waveIO as wp
import peakProcessing as PP
import errorHandler as EH

try:
  import genSpecSines_C as GS
  import twm_C as fd
except ImportError:
  import genSpecSines as GS
  import twm as fd
  EH.printWarning(1)


def hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm=10):
  # Analysis/synthesis of a sound using the harmonic plus stochastic model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # returns y: output sound, yh: harmonic component, yst: stochastic component

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
  ystw = np.zeros(Ns)                                            # initialize output sound frame
  yh = np.zeros(x.size)                                         # initialize output array
  yst = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]           # synthesis window for harmonic component
  sws = H*hanning(Ns)/2                                         # synthesis window for stochastic
  while pin<pend:  
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                                      # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                           # compute dft
    ploc = PP.peakDetection(mX, hN, t)                           # find peaks                
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)          # refine peak values
    f0 = fd.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
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
    hloc[:hi] = (hloc[:hi]!=0) * (hloc[:hi]*Ns/N)                # synth. locs
    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                          # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                   # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]                           
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
  #-----synthesis-----
    Yh = GS.genSpecSines(hloc[:hi], hmag, hphase, Ns)            # generate spec sines of harmonic component          
    Yr = X2-Yh;                                                  # get the residual complex spectrum
    mYr = 20 * np.log10(abs(Yr[:hNs]))                           # magnitude spectrum of residual
    mYrenv = resample(np.maximum(-200, mYr), mYr.size*stocf)     # decimate the magnitude spectrum and avoid -Inf                     
    mYst = resample(mYrenv, hNs)                                 # interpolate to original size
    mYst = 10**(mYst/20)                                         # dB to linear magnitude  
    pYst = 2*np.pi*np.random.rand(hNs)                           # generate phase random values
    Yst = np.zeros(Ns, dtype = complex)
    Yst[:hNs] = mYst * np.exp(1j*pYst)                           # generate positive freq.
    Yst[hNs+1:] = mYst[:0:-1] * np.exp(-1j*pYst[:0:-1])          # generate negative freq.
    
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Yh))                                # inverse FFT of harmonic spectrum
    yhw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    yhw[hNs-1:] = fftbuffer[:hNs+1] 

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Yst))                                # inverse FFT of stochastic spectrum
    ystw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    ystw[hNs-1:] = fftbuffer[:hNs+1]

    yh[ri:ri+Ns] += sw*yhw                                       # overlap-add for sines
    yst[ri:ri+Ns] += sws*ystw                                    # overlap-add for stochastic
    pin += H                                                     # advance sound pointer
  
  y = yh+yst                                                     # sum of harmonic and stochastic components
  return y, yh, yst

def defaultTest():
    str_time = time.time()
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
    w = np.blackman(801)
    N = 1024
    t = -90
    nH = 50
    minf0 = 350
    maxf0 = 700
    f0et = 10
    maxhd = 0.2
    stocf = 0.2
    maxnpeaksTwm = 5
    y, yh, yst = hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
    print "time taken for computation " + str(time.time()-str_time)
  
if __name__ == '__main__':
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(801)
  N = 1024
  t = -90
  nH = 50
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxhd = 0.2
  stocf = 0.2
  maxnpeaksTwm = 5
  y, yh, yst = hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)

  wp.play(y, fs)
  wp.play(yh, fs)
  wp.play(yst, fs)
