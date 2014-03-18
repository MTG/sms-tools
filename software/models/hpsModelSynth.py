import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time
import hpsModelAnal

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import waveIO as WIO
import hpsModelAnal as HPS
import errorHandler as EH

try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  EH.printWarning(1)

def hpsModelSynth(hfreq, hmag, hphase, mYst, N, H, fs):
  # Synthesis of a sound using the harmonic plus stochastic model
  # hfreq: harmonic frequencies, hmag:harmonic amplitudes, mYst: stochastic envelope
  # Ns: synthesis FFT size, H: hop size, fs: sampling rate 
  # y: output sound, yh: harmonic component, yst: stochastic component
  hN = N/2                                                  # half of FFT size for synthesis
  L = hfreq[:,0].size                                       # number of frames
  nH = hfreq[0,:].size                                      # number of harmonics
  pout = 0                                                  # initialize output sound pointer         
  ysize = H*(L+4)                                           # output sound size
  yhw = np.zeros(N)                                        # initialize output sound frame
  ysw = np.zeros(N)                                        # initialize output sound frame
  yh = np.zeros(ysize)                                      # initialize output array
  yst = np.zeros(ysize)                                     # initialize output array
  sw = np.zeros(N)     
  ow = triang(2*H)                                          # overlapping window
  sw[hN-H:hN+H] = ow      
  bh = blackmanharris(N)                                   # synthesis window
  bh = bh / sum(bh)                                         # normalize synthesis window
  wr = bh                                                   # window for residual
  sw[hN-H:hN+H] = sw[hN-H:hN+H] / bh[hN-H:hN+H]             # synthesis window for harmonic component
  sws = H*hanning(N)/2                                      # synthesis window for stochastic component
  lastyhfreq = hfreq[0,:]                                   # initialize synthesis harmonic locations
  yhphase = 2*np.pi*np.random.rand(nH)                      # initialize synthesis harmonic phases     
  for l in range(L):
    yhfreq = hfreq[l,:]                                     # synthesis harmonics frequencies
    yhmag = hmag[l,:]                                       # synthesis harmonic amplitudes
    mYrenv = mYst[l,:]                                      # synthesis residual envelope
    if (hphase.size > 0):
      yhphase = hphase[l,:] 
    else:
      yhphase += (np.pi*(lastyhfreq+yhfreq)/fs)*H             # propagate phases
    lastyhfreq = yhfreq
    Yh = GS.genSpecSines(N*yhfreq/fs, yhmag, yhphase, N)   # generate spec sines 
    mYs = resample(mYrenv, hN)                              # interpolate to original size
    mYs = 10**(mYs/20)                                      # dB to linear magnitude  
    pYs = 2*np.pi*np.random.rand(hN)                        # generate phase random values
    Ys = np.zeros(N, dtype = complex)
    Ys[:hN] = mYs * np.exp(1j*pYs)                         # generate positive freq.
    Ys[hN+1:] = mYs[:0:-1] * np.exp(-1j*pYs[:0:-1])        # generate negative freq.
    fftbuffer = np.zeros(N)
    fftbuffer = np.real(ifft(Yh))                           # inverse FFT of harm spectrum
    yhw[:hN-1] = fftbuffer[hN+1:]                         # undo zer-phase window
    yhw[hN-1:] = fftbuffer[:hN+1] 
    fftbuffer = np.zeros(N)
    fftbuffer = np.real(ifft(Ys))                           # inverse FFT of stochastic approximation spectrum
    ysw[:hN-1] = fftbuffer[hN+1:]                           # undo zero-phase window
    ysw[hN-1:] = fftbuffer[:hN+1]
    yh[pout:pout+N] += sw*yhw                               # overlap-add for sines
    yst[pout:pout+N] += sws*ysw                             # overlap-add for stoch
    pout += H                                               # advance sound pointer
  y = yh+yst                                                # sum harmonic and stochastic components
  return y, yh, yst

if __name__ == '__main__':
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase-short.wav'))
  w = np.blackman(601)
  N = 1024
  t = -100
  nH = 100
  minf0 = 350
  maxf0 = 700
  f0et = 5
  maxnpeaksTwm = 5
  minSineDur = .1
  harmDevSlope = 0.01
  Ns = 512
  H = Ns/4
  stocf = .2
  hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur, Ns, stocf)
  y, yh, yst = hpsModelSynth(hfreq, hmag, hphase, mYst, Ns, H, fs)
  WIO.play(y, fs)
  WIO.play(yh, fs)
  WIO.play(yst, fs)


