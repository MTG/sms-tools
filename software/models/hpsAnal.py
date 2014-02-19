import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal
import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH

try:
  import genSpecSines_C as GS
  import twm_C as TWM
except ImportError:
  import genSpecSines as GS
  import twm as TWM
  EH.printWarning(1)
  

def hpsAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm=10):
  # Analysis of a sound using the harmonic plus stochastic model, prepared for transformations
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # xhloc: harmonic locations, xhmag:harmonic amplitudes, xmXrenv: residual envelope, Ns: residual FFT size

  hN = N/2                                                      # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
  Ns = 512                                                      # FFT size for synthesis
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2                                                    # half of FFT size for synthesis
  pin = hM1                                                     # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  w = w / sum(w)                                                # normalize analysis window
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual

  while pin<pend:          
    x1 = x[pin-hM1:pin+hM2]                                      # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                           # compute dft
    ploc = PP.peakDetection(mX, hN, t)                           # detect spectral peaks
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)          # refine peak values 
    f0 = TWM.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
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
    hloc[:hi] = (hloc[:hi]!=0) * (hloc[:hi]*Ns/N)                # synth. locs
    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                         # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                  # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]                            
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
    Xh = GS.genSpecSines(hloc, hmag, hphase, Ns)                 # generate sines
    Xr = X2-Xh                                                   # get the residual complex spectrum
    mXr = 20 * np.log10(abs(Xr[:hNs]))                           # magnitude spectrum of residual
    mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)     # decimate the mag spectrum                                                  
    if pin == hM1: 
      xhloc = np.array([hloc])
      xhmag = np.array([hmag])
      xmXrenv = np.array([mXrenv])
    else:
      xhloc = np.vstack((xhloc,np.array([hloc])))
      xhmag = np.vstack((xhmag, np.array([hmag])))
      xmXrenv = np.vstack((xmXrenv, np.array([mXrenv])))
    pin += H                                                     # advance sound pointer
  return xhloc, xhmag, xmXrenv, Ns, H

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(801)
  N = 1024
  t = -90
  nH = 50
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxhd = 0.2
  stocf = .2
  maxnpeaksTwm = 5
  hloc, hmag, mXrenv, Ns, H = hpsAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
  print "time taken for computation " + str(time.time()-str_time)  
  

if __name__ == '__main__':
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(801)
  N = 1024
  t = -90
  nH = 50
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxhd = 0.2
  stocf = .2
  maxnpeaksTwm = 5
  hloc, hmag, mXrenv, Ns, H = hpsAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
  numFrames = int(hloc[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)
  binFreq = np.arange(stocf*Ns/2)*float(fs)/(stocf*Ns)
  plt.pcolormesh(frmTime, binFreq, np.transpose(mXrenv))
  plt.plot(frmTime, hloc*float(fs)/Ns, 'x', ms=3, alpha=.4)
  plt.xlabel('Time(s)')
  plt.ylabel('Frequency(Hz)')
  plt.autoscale(tight=True)
  plt.title('harmonic + stochastic components')
  plt.show()



