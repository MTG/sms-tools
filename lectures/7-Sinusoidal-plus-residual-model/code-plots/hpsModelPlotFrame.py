import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../code/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../../code/basicFunctions_C/'))

import smsF0DetectionTwm as fd
import smsWavplayer as wp
import smsPeakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

  
def hprModelFrame(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf):
  hN = N/2                                                      # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  yhw = np.zeros(Ns)                                            # initialize output sound frame
  yrw = np.zeros(Ns)                                            # initialize output sound frame
  yh = np.zeros(x.size)                                         # initialize output array
  yr = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]

  #-----analysis-----             
  xw = x[pin-hM1:pin+hM2] * w                                  # window the input sound
  fftbuffer = np.zeros(N)                                      # reset buffer
  fftbuffer[:hM1] = xw[hM2:]                                   # zero-phase window in fftbuffer
  fftbuffer[N-hM2:] = xw[:hM2]                           
  X = fft(fftbuffer)                                           # compute FFT
  mX = 20 * np.log10(abs(X[:hN]))                              # magnitude spectrum of positive frequencies
  ploc = PP.peakDetection(mX, hN, t)                
  pX = np.unwrap(np.angle(X[:hN]))                             # unwrapped phase spect. of positive freq.    
  iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)          # refine peak values
    
  f0 = fd.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0)  # find f0
  hloc = np.zeros(nH)                                          # initialize harmonic locations
  hmag = np.zeros(nH)-100                                      # initialize harmonic magnitudes
  hphase = np.zeros(nH)                                        # initialize harmonic phases
  hf = (f0>0)*(f0*np.arange(1, nH+1))                          # initialize harmonic frequencies
  hi = 0                                                       # initialize harmonic index
  npeaks = ploc.size;                                          # number of peaks found
    
  while f0>0 and hi<nH and hf[hi]<fs/2 :                       # find harmonic peaks
      dev = min(abs(iploc/N*fs - hf[hi]))
      pei = np.argmin(abs(iploc/N*fs - hf[hi]))                  # closest peak
      if ( hi==0 or not any(hloc[:hi]==iploc[pei]) ) and dev<maxhd*hf[hi] :
        hloc[hi] = iploc[pei]                                    # harmonic locations
        hmag[hi] = ipmag[pei]                                    # harmonic magnitudes
        hphase[hi] = ipphase[pei]                                # harmonic phases
      hi += 1                                                    # increase harmonic index
    
  hlocN = hloc
  hloc[:hi] = (hloc[:hi]!=0) * (hloc[:hi]*Ns/N)                # synth. locs
  ri = pin-hNs-1                                               # input sound pointer for residual analysis
  xr = x[ri:ri+Ns]*wr                                          # window the input sound                                       
  fftbuffer = np.zeros(Ns)                                     # reset buffer
  fftbuffer[:hNs] = xr[hNs:]                                   # zero-phase window in fftbuffer
  fftbuffer[hNs:] = xr[:hNs]                           
  Xr = fft(fftbuffer)                                          # compute FFT for residual analysis
  
  #-----synthesis-----
  Yh = GS.genSpecSines(hloc[:hi], hmag, hphase, Ns)            # generate spec sines of harmonic component          
  mYh = 20 * np.log10(abs(Yh[:hNs]))
  pYh = np.unwrap(np.angle(Yh[:hNs])) 
  Yr = Xr-Yh;                                                  # get the residual complex spectrum
  mXr = 20 * np.log10(abs(Xr[:hNs]))
  pXr = np.unwrap(np.angle(Xr[:hNs])) 
  mYr = 20 * np.log10(abs(Yr[:hNs]))
  pYr = np.unwrap(np.angle(Yr[:hNs])) 
  mYrenv = resample(np.maximum(-200, mYr), mYr.size*stocf)
  mYs = resample(mYrenv, hNs)
 
  fftbuffer = np.zeros(Ns)
  fftbuffer = np.real(ifft(Yh))                                # inverse FFT of harmonic spectrum
  yhw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
  yhw[hNs-1:] = fftbuffer[:hNs+1] 
    
  fftbuffer = np.zeros(Ns)
  fftbuffer = np.real(ifft(Yr))                                # inverse FFT of residual spectrum
  yrw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
  yrw[hNs-1:] = fftbuffer[:hNs+1]
    
  yh[ri:ri+Ns] += sw*yhw                                       # overlap-add for sines
  yr[ri:ri+Ns] += sw*yrw                                       # overlap-add for residual
  
  y = yh+yr                                                      # sum of harmonic and residual components
  return mX, hloc, hmag, mXr, mYh, mYr, mYs

  
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../../../sounds/flute-A4.wav')
  w = np.blackman(601)
  N = 2048
  Ns = 512
  t = -90
  nH = 30
  minf0 = 350
  maxf0 = 600
  f0et = 10
  maxhd = 0.2
  stocf = .2
  maxFreq = 10000.0
  lastbin = N*maxFreq/fs
  first = 40000
  last = first+w.size
  mX, hloc, hmag, mXr, mYh, mYr, mYs = hprModelFrame(x[first:last], fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf)

  plt.figure(1)
  plt.subplot(2,1,1)
  plt.plot(np.arange(0, fs/2.0, fs/float(N)), mX, 'k')
  plt.plot((hloc/Ns)*fs, hmag, 'r*')
  plt.axis([0, maxFreq, -100, max(mX)+1])
  plt.title('mX')

  plt.subplot(2,1,2)
  plt.plot(np.arange(0, fs/2.0, fs/float(Ns)), mXr, 'k')
  plt.plot(np.arange(0, fs/2.0, fs/float(Ns)), mYh, 'b')
  # plt.plot((hloc/Ns)*fs, hmag, 'r*')
  plt.plot(np.arange(0, fs/2.0, fs/float(Ns)), mYr, 'g')
  plt.plot(np.arange(0, fs/2.0, fs/float(Ns)), mYs, 'r')
  plt.axis([0, maxFreq, -100, max(mYh)+1])
  plt.title('mXr (black), mYh (blue), mYr (green), mYs (red)')
  plt.show()

