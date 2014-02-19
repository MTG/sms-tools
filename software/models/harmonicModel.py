import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import math

import sys, os, functools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import twm as fd
import waveIO as wp
import peakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import genSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm=10):
  # Analysis/synthesis of a sound using the sinusoidal harmonic model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # maxnpeaksTwm: maximum number of peaks used for F0 detection
  # yh: harmonic component, yr: residual component
  # returns y: output array sound

  
  hN = N/2                                                      # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  yh = np.zeros(Ns)                                             # initialize output sound frame
  y = np.zeros(x.size)                                          # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)                                             # initialize synthesis window
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  while pin<pend:             
  #-----analysis-----             
    xw = x[pin-hM1:pin+hM2] * w                                  # window the input sound
    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM1] = xw[hM2:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM2:] = xw[:hM2]                           
    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
    ploc = PP.peakDetection(mX, hN, t)                           # detect peak locations
    pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.     
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
    hloc = (hloc!=0) * (hloc*Ns/N)                               # synth. locs
  #-----synthesis-----
    Yh = GS.genSpecSines(hloc, hmag, hphase, Ns)                 # generate spec sines          
    fftbuffer = np.real( ifft(Yh) )                              # inverse FFT
    yh[:hNs-1] = fftbuffer[hNs+1:]                               # undo zero-phase window
    yh[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yh                                  # overlap-add
    pin += H                                                     # advance sound pointer
  return y

def defaultTest():
    str_time = time.time()    
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
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
    (fs, x) = wp.wavread('../../sounds/sax-phrase-short.wav')
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
    wp.play(y, fs)