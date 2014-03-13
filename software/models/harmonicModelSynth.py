import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import harmonicModelAnal as HA
import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH

try:
  import genSpecSines_C as GS
  import twm_C as fd
except ImportError:
  import genSpecSines as GS
  import twm as fd
  EH.printWarning(1)

def harmonicModelSynth(hfreq, hmag, hphase, fs):
  # Synthesis of a sound using the sinusoidal harmonic model
  # hfreq, hmag, hphase: harmonic frequencies, magnitudes and phases
  # returns y: output array sound 
  Ns = 512                                                # FFT size for synthesis (even)
  H = Ns/4                                                # Hop size used for analysis and synthesis
  hNs = Ns/2      
  l = pout = 0                                            # initialize output sound pointer 
  L = hfreq[:,0].size                                     # number of frames   
  ysize = H*(L+3)                                         # output sound size
  yh = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(ysize)                                     # initialize output array
  sw = np.zeros(Ns)                                       # initialize synthesis window
  ow = triang(2*H)                                        # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                 # synthesis window
  bh = bh / sum(bh)                                       # normalize synthesis window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # window for overlap-add
  while l<L:
    yhfreq = hfreq[l,:]                                   # synthesis harmonics frequencies
    yhmag = hmag[l,:]                                     # synthesis harmonic amplitudes
    yhphase = hphase[l,:]                                 # synthesis harmonic phases
    Yh = GS.genSpecSines(Ns*yhfreq/fs, yhmag, yhphase, Ns)   # generate spec sines          
    fftbuffer = np.real(ifft(Yh))                         # inverse FFT
    yh[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yh[hNs-1:] = fftbuffer[:hNs+1] 
    y[pout:pout+Ns] += sw*yh                              # overlap-add
    l += 1                                                # advance frame index
    pout += H                                             # advance sound pointer
  return y

if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../sounds/vignesh.wav')
  w = np.blackman(1201)
  N = 2048
  t = -90
  nH = 100
  minf0 = 130
  maxf0 = 300
  f0et = 7
  maxnpeaksTwm = 4
  Ns = 512
  H = Ns/4
  minSineDur = .1
  harmDevSlope = 0.01

  hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
  y = harmonicModelSynth(hfreq, hmag, hphase, fs)
  WIO.play(y, fs)

