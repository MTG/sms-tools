import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal as DF
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
  
  
def hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm=10):
  # Analysis/synthesis of a sound using the harmonic plus residual model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # returns y: output sound, yh: harmonic component, xr: residual component

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
  xrw = np.zeros(Ns)                                            # initialize output sound frame
  yh = np.zeros(x.size)                                         # initialize output array
  xr = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  hfreqp = []
  while pin<pend:  
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                          # select frame
    mX, pX = DF.dftAnal(x1, w, N)                    # compute dft
    ploc = PP.peakDetection(mX, hN, t)               # find peaks 
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)  # refine peak values
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
    #-----synthesis-----
    Yh = GS.genSpecSines(Ns*hfreq/fs, hmag, hphase, Ns)    # generate sines
    Xr = X2-Yh                                      # get the residual complex spectrum                       
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Yh))                   # inverse FFT of harmonic spectrum
    yhw[:hNs-1] = fftbuffer[hNs+1:]                 # undo zero-phase window
    yhw[hNs-1:] = fftbuffer[:hNs+1] 
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Xr))                   # inverse FFT of residual spectrum
    xrw[:hNs-1] = fftbuffer[hNs+1:]                 # undo zero-phase window
    xrw[hNs-1:] = fftbuffer[:hNs+1]
    yh[ri:ri+Ns] += sw*yhw                          # overlap-add for sines
    xr[ri:ri+Ns] += sw*xrw                          # overlap-add for residual
    pin += H                                        # advance sound pointer
  y = yh+xr                                         # sum of harmonic and residual components
  return y, yh, xr

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(701)
  N = 1024
  t = -90
  nH = 60
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxnpeaksTwm = 5
  y, yh, xr = hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)
  print "time taken for computation " + str(time.time()-str_time)
  
if __name__ == '__main__':
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(701)
  N = 1024
  t = -90
  nH = 60
  minf0 = 350
  maxf0 = 700
  f0et = 10
  maxnpeaksTwm = 5
  y, yh, xr = hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)
  WIO.play(y, fs)
  WIO.play(yh, fs)
  WIO.play(xr, fs)
