import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))

import dftAnal, stftAnal
import sineTracking as ST
import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH

def sineModelAnal(x, fs, w, N, H, t, maxnSines = 100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
  # Analysis of a sound using the sinusoidal model
  # x: input array sound, w: analysis window, N: size of complex spectrum,
  # H: hop-size, t: threshold in negative dB
  # maxnSines: maximum number of sines per frame
  # minSineDur: minimum duration of sines in seconds
  # freqDevOffset: minimum frequency deviation at 0Hz 
  # freqDevSlope: slope increase of minimum frequency deviation
  # returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoids
  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
  pin = hM1                                               # initialize sound pointer in middle of analysis window       
  pend = x.size - hM1                                     # last sample to start a frame
  w = w / sum(w)                                          # normalize analysis window
  tfreq = np.array([])
  while pin<pend:                                         # while input sound pointer is within sound            
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft
    ploc = PP.peakDetection(mX, hN, t)                    # detect locations of peaks
    pmag = mX[ploc]                                       # get the magnitude of the peaks
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
    ipfreq = fs*iploc/float(N)
    tfreq, tmag, tphase = ST.sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
    tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))
    tmag = np.resize(tmag, min(maxnSines, tmag.size))
    tphase = np.resize(tphase, min(maxnSines, tphase.size))
    jtfreq = np.zeros(maxnSines) 
    jtmag = np.zeros(maxnSines)  
    jtphase = np.zeros(maxnSines)    
    jtfreq[:tfreq.size]=tfreq 
    jtmag[:tmag.size]=tmag
    jtphase[:tphase.size]=tphase 
    if pin == hM1:
      xtfreq = jtfreq 
      xtmag = jtmag
      xtphase = jtphase
    else:
      xtfreq = np.vstack((xtfreq, jtfreq))
      xtmag = np.vstack((xtmag, jtmag))
      xtphase = np.vstack((xtphase, jtphase))
    pin += H
  xtfreq = ST.cleaningSineTracks(xtfreq, round(fs*minSineDur/H))
  return xtfreq, xtmag, xtphase

def defaultTest():
  str_time = time.time()
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.hamming(2001)
  N = 2048
  H = 200
  t = -80
  minSineDur = .02
  maxnSines = 150
  freqDevOffset = 10
  freqDevSlope = 0.001
  mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
  tfreq, tmag, tphase = sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
  print "time taken for computation " + str(time.time()-str_time)  

if __name__ == '__main__':
  (fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.hamming(2001)
  N = 2048
  H = 200
  t = -80
  minSineDur = .02
  maxnSines = 150
  freqDevOffset = 10
  freqDevSlope = 0.001
  mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
  tfreq, tmag, tphase = sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

  maxplotfreq = 5000.0
  maxplotbin = int(N*maxplotfreq/fs)
  numFrames = int(mX[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
  plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
  plt.autoscale(tight=True)
  
  tracks = tfreq*np.less(tfreq, maxplotfreq)
  tracks[tracks<=0] = np.nan
  plt.plot(frmTime, tracks, color='k')
  plt.autoscale(tight=True)
  plt.title('sinusoidal tracks on spectrogram')
  plt.show()