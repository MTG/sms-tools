import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time
import hprModelAnal

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as WIO
import errorHandler as EH

try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  EH.printWarning(1)

def hprModelSynth(hfreq, hmag, hphase, xr, Ns, H, fs):
  # Synthesis of a sound using the harmonic plus stochastic model
  # hfreq: harmonic frequencies, hmag:harmonic amplitudes, hmag:harmonic phases, xr: residual component
  # Ns: synthesis FFT size, H: hop size, fs: sampling rate 
  # y: output sound, yh: harmonic component
  hNs = Ns/2                                                # half of FFT size for synthesis
  l = 0                                                     # frame index
  L = hfreq[:,0].size                                       # number of frames
  nH = hfreq[0,:].size                                      # number of harmonics
  pout = 0                                                  # initialize output sound pointer         
  ysize = H*(L+3)                                           # output sound size
  yhw = np.zeros(Ns)                                        # initialize output sound frame
  yh = np.zeros(ysize)                                      # initialize output array
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                          # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                   # synthesis window
  bh = bh / sum(bh)                                         # normalize synthesis window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]       # synthesis window for harmonic component  
  while l<L:
    yhfreq = hfreq[l,:]                                     # synthesis harmonics frequencies
    yhmag = hmag[l,:]                                       # synthesis harmonic amplitudes
    yhphase = hphase[l,:]                                   # synthesis harmonic phases
    Yh = GS.genSpecSines(Ns*yhfreq/fs, yhmag, yhphase, Ns)  # generate spec sines 
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Yh))                           # inverse FFT of harm spectrum
    yhw[:hNs-1] = fftbuffer[hNs+1:]                         # undo zer-phase window
    yhw[hNs-1:] = fftbuffer[:hNs+1] 
    yh[pout:pout+Ns] += sw*yhw                              # overlap-add for sines
    l += 1                                                  # advance frame pointer
    pout += H                                               # advance sound pointer
  y = yh+yr                                                 # sum harmonic and residual components
  return y, yh

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
  stocf = 0.2
  maxnpeaksTwm = 5
  hfreq, hmag, hphase, xr, H = hprModelAnal.hprModelAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)
  y, yh, yst = hpsModelSynth(hfreq, hmag, hphase, xr, Ns, H, fs)
  WIO.play(y, fs)
  WIO.play(yh, fs)
  WIO.play(xr, fs)


