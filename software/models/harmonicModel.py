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
import harmonicDetection as HD
import dftAnal

try:
  import genSpecSines_C as GS
  import twm_C as TWM
except ImportError:
  import genSpecSines as GS
  import twm as TWM
  EH.printWarning(1)

def harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm=10):
  # Analysis/synthesis of a sound using the sinusoidal harmonic model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
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
  hfreqp = []
  while pin<pend:             
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft
    ploc = PP.peakDetection(mX, hN, t)                    # detect peak locations     
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0 = TWM.f0DetectionTwm(ipfreq, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)  # find f0
    hfreq, hmag, hphase = HD.harmonicDetection(ipfreq, ipmag, ipphase, f0, nH, hfreqp, fs) # find harmonics
    hfreqp = hfreq
  #-----synthesis-----
    Yh = GS.genSpecSines(Ns*hfreq/fs, hmag, hphase, Ns)          # generate spec sines          
    fftbuffer = np.real(ifft(Yh))                         # inverse FFT
    yh[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yh[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yh                           # overlap-add
    pin += H                                              # advance sound pointer
  return y

def defaultTest():
    str_time = time.time()    
    (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/vignesh.wav'))
    w = np.blackman(701)
    N = 1024
    t = -80
    nH = 30
    minf0 = 200
    maxf0 = 300
    f0et = 5
    maxhd = 0.2
    maxnpeaksTwm = 5
    y = harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)    
    print "time taken for computation " + str(time.time()-str_time)


if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/vignesh.wav')
  w = np.blackman(801)
  N = 2048
  t = -90
  nH = 40
  minf0 = 130
  maxf0 = 300
  f0et = 7
  maxnpeaksTwm = 4
  Ns = 512
  H = Ns/4
  y = harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)
  WIO.play(y, fs)
