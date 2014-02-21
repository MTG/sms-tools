import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import math
import sys, os, functools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH
import dftAnal

try:
  import genSpecSines_C as GS
  import twm_C as TWM
except ImportError:
  import genSpecSines as GS
  import twm as TWM
  EH.printWarning(1)

def harmonicDetection (ploc, pmag, pphase, f0, nH, maxhd, plocp, pmagp):
  # detection of the harmonics from a set of spectral peaks
  # ploc: peak locations, pmag: peak magnitudes, pphase: peak phases
  # f0: fundamental frequency, nH: number of harmonics,
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # plocp: peak locations of previous frame, pmagp: peak magnitude of previous frame,
  # returns hloc: harmonic locations, hmag: harmonic magnitudes, hphase: harmonic phases
  hloc = np.zeros(nH)                                         # initialize harmonic locations
  hmag = np.zeros(nH)-100                                     # initialize harmonic magnitudes
  hphase = np.zeros(nH)                                       # initialize harmonic phases
  hf = (f0>0)*(f0*np.arange(1, nH+1))                         # initialize harmonic frequencies
  hi = 0                                                      # initialize harmonic index
  npeaks = ploc.size                                          # number of peaks found
  while f0>0 and hi<nH and hf[hi]<fs/2:                       # find harmonic peaks
    dev = min(abs(ploc/N*fs - hf[hi]))
    pei = np.argmin(abs(ploc/N*fs - hf[hi]))                  # closest peak
    if ( hi==0 or not any(hloc[:hi]==ploc[pei]) ) and dev<maxhd*hf[hi] :
      hloc[hi] = ploc[pei]                                    # harmonic locations
      hmag[hi] = pmag[pei]                                    # harmonic magnitudes
      hphase[hi] = pphase[pei]                                # harmonic phases
    hi += 1                                                   # increase harmonic index
  return hloc, hmag, hphase

def harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm=10):
  # Analysis/synthesis of a sound using the sinusoidal harmonic model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # returns y: output array sound
  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  Ns = 512                                                # FFT size for synthesis (even)
  H = Ns/4                                                # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window          
  pend = x.size - max(hNs, hM1)                           # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yh = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window
  sw = np.zeros(Ns)                                       # initialize synthesis window
  ow = triang(2*H)                                        # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                 # synthesis window
  bh = bh / sum(bh)                                       # normalize synthesis window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # window for overlap-add
  hlocp = hmagp = []
  while pin<pend:             
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft
    ploc = PP.peakDetection(mX, hN, t)                    # detect peak locations     
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values
    f0 = TWM.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
    hloc, hmag, hphase = harmonicDetection(iploc, ipmag, ipphase, f0, nH, maxhd, hlocp, hmagp) # find harmonics
    hlocp = hloc
    hmagp = hmagp
    hloc = (hloc!=0) * (hloc*Ns/N)                        # synth. locs
  #-----synthesis-----
    Yh = GS.genSpecSines(hloc, hmag, hphase, Ns)          # generate spec sines          
    fftbuffer = np.real(ifft(Yh))                         # inverse FFT
    yh[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yh[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yh                           # overlap-add
    pin += H                                              # advance sound pointer
  return y

def defaultTest():
    str_time = time.time()    
    (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
    w = np.blackman(701)
    N = 1024
    t = -80
    nH = 30
    minf0 = 400
    maxf0 = 700
    f0et = 5
    maxhd = 0.2
    y = harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd)    
    print "time taken for computation " + str(time.time()-str_time)


if __name__ == '__main__':
    (fs, x) = WIO.wavread('../../sounds/sax-phrase-short.wav')
    w = np.blackman(901)
    N = 1024
    t = -90
    nH = 40
    minf0 = 350
    maxf0 = 700
    f0et = 10
    maxhd = 0.2
    maxnpeaksTwm = 5
    y = harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm)
    WIO.play(y, fs)
