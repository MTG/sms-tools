import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal, stftAnal
import waveIO as wp
import peakProcessing as PP

try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"

def sineModelAnal(x, fs, w, N, H, t):
  # Analysis of a sound using the sinusoidal model
  # x: input array sound, w: analysis window, N: size of complex spectrum,
  # H: hop-size, t: threshold in negative dB 
  # returns xploc: peak locations, xpmag: peak magnitudes, xpphase: peak phases
  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  maxnpeaks=150                                           # set a maximum number of peaks
  pin = hM1                                               # initialize sound pointer in middle of analysis window       
  pend = x.size - hM1                                     # last sample to start a frame
  w = w / sum(w)                                          # normalize analysis window
  while pin<pend:                                         # while input sound pointer is within sound            
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft
    ploc = PP.peakDetection(mX, hN, t)                    # detect locations of peaks
    pmag = mX[ploc]                                       # get the magnitude of the peaks
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
    npeaks = min(maxnpeaks,iploc.size)                    # number of peaks of current frame
    jploc = np.zeros(maxnpeaks)   
    jploc[:npeaks]=iploc[:npeaks] 
    jpmag=np.zeros(maxnpeaks) 
    jpmag[:npeaks]=ipmag[:npeaks]  
    jpphase=np.zeros(maxnpeaks) 
    jpphase[:npeaks]=ipphase[:npeaks]  
    if pin == hM1:
      xploc = jploc 
      xpmag = jpmag
      xpphase = jpphase
    else:
      xploc = np.vstack((xploc, jploc))
      xpmag = np.vstack((xpmag, jpmag))
      xpphase = np.vstack((xpphase, jpphase))
    pin += H                                              # advance sound pointer
  return xploc, xpmag, xpphase

def defaultTest():
  str_time = time.time()
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.hamming(2001)
  N = 2048
  H = 1000
  t = -80
  ploc, pmag, pphase = sineModelAnal(x, fs, w, N, H, t)
  print "time taken for computation " + str(time.time()-str_time)  
  
# example call of sineModelAnal function
if __name__ == '__main__':
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.hamming(2001)
  N = 2048
  H = 1000
  t = -80
  mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
  ploc, pmag, pphase = sineModelAnal(x, fs, w, N, H, t)

  maxplotbin = int(N*800.0/fs)
  numFrames = int(mX[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
  plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
  plt.autoscale(tight=True)
  
  peaks = ploc*np.less(ploc,maxplotbin)*float(fs)/N
  peaks[peaks==0] = np.nan
  plt.plot(frmTime, peaks, 'x', color='k')
  plt.autoscale(tight=True)
  plt.title('spectral peaks on spectrogram')
  plt.show()
