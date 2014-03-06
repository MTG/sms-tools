import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal, stftAnal
import waveIO as WIO
import peakProcessing as PP
import harmonicDetection as HD
import errorHandler as EH

try:
  import genSpecSines_C as GS
  import twm_C as TWM
except ImportError:
  import genSpecSines as GS
  import twm as TWM
  EH.printWarning(1)
  

def hprModelAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm=10):
  # Analysis of a sound using the harmonic plus stochastic model, prepared for transformations
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # returns xhfreq: harmonic locations, xhmag: harmonic amplitudes, xhphase: harmonic phases, 
  # xr: residual component, H: hop size
  hN = N/2                                         # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))              # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                  # half analysis window size by floor
  Ns = 512                                         # FFT size for synthesis
  H = Ns/4                                         # Hop size used for analysis and synthesis
  hNs = Ns/2                                       # half of FFT size for synthesis
  pin = hM1                                        # initialize sound pointer in middle of anal window          
  pend = x.size - max(hNs, hM1)                    # last sample to start a frame
  w = w / sum(w)                                   # normalize analysis window
  bh = blackmanharris(Ns)                          # synthesis window
  bh = bh / sum(bh)                                # normalize synthesis window
  wr = bh                                          # window for residual
  sw = np.zeros(Ns)
  ow = triang(2*H)                                 # overlapping window
  sw[hNs-H:hNs+H] = ow      
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  xr = np.zeros(x.size)                            # initialize output array
  xrw = np.zeros(Ns)                               # initialize output sound frame
  hfreqp = []
  while pin<pend:          
    x1 = x[pin-hM1:pin+hM2]                        # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)             # compute dft
    ploc = PP.peakDetection(mX, hN, t)             # detect spectral peaks
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values 
    ipfreq = fs * iploc/N
    f0 = TWM.f0DetectionTwm(ipfreq, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
    hfreq, hmag, hphase = HD.harmonicDetection(ipfreq, ipmag, ipphase, f0, nH, hfreqp, fs) # find harmonics
    hfreqp = hfreq
    ri = pin-hNs-1                                  # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                            # window the input sound                     
    fftbuffer = np.zeros(Ns)                        # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                     # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]                     
    X2 = fft(fftbuffer)                             # compute FFT for residual analysis
    Xh = GS.genSpecSines(Ns*hfreq/fs, hmag, hphase, Ns)    # generate sines
    Xr = X2-Xh                                      # get the residual complex spectrum                       
    if pin == hM1: 
      xhfreq = np.array([hfreq])
      xhmag = np.array([hmag])
      xhphase = np.array([hphase])
    else:
      xhfreq = np.vstack((xhfreq,np.array([hfreq])))
      xhmag = np.vstack((xhmag, np.array([hmag])))
      xhphase = np.vstack((xhphase, np.array([hphase])))
    fftbuffer = np.real(ifft(Xr))                                # inverse FFT of residual spectrum
    xrw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    xrw[hNs-1:] = fftbuffer[:hNs+1]
    xr[ri:ri+Ns] += sw*xrw                                       # overlap-add for residual
    pin += H                                                     # advance sound pointer
  return xhfreq, xhmag, xhphase, xr, H

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(801)
  N = 1024
  t = -100
  nH = 70
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxnpeaksTwm = 5
  hfreq, hmag, hphase, xr, H = hprModelAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)
  print "time taken for computation " + str(time.time()-str_time)  
  

if __name__ == '__main__':
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(551)
  N = 1024
  t = -100
  nH = 70
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxnpeaksTwm = 5
  hfreq, hmag, hphase, xr, H = hprModelAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)

  mXr, pXr = stftAnal.stftAnal(xr, fs, hamming(H*2), H*2, H)
  numFrames = int(mXr[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(H)*float(fs)/(H*2)                       
  plt.pcolormesh(frmTime, binFreq, np.transpose(mXr))
  plt.autoscale(tight=True)

  hfreq[hfreq==0] = np.nan
  numFrames = int(hfreq[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs) 
  plt.plot(frmTime, hfreq, color='k', ms=3, alpha=1)
  plt.xlabel('Time(s)')
  plt.ylabel('Frequency(Hz)')
  plt.autoscale(tight=True)
  plt.title('harmonic + residual components')
  plt.show()
