import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
from scipy.interpolate import interp1d
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import hpsAnal, hpsSynth
import smsWavplayer as wp

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def hpsTimeScale(hloc, hmag, stocEnv, inTime, outTime):
  # Synthesis of a sound using the harmonic plus stochastic model
  # hloc: harmonic locations, hmag:harmonic amplitudes, mXrenv: residual envelope
  # Ns: synthesis FFT size, H: hop size, fs: sampling rate, inTime: input time, outTime: output time
  # yhloc: harmonic locations, yhmag:harmonic amplitudes, stocEnv: residual envelope
  l = 0                                                        # frame index
  L = hloc[:,0].size                                           # number of analysis frames
  nH = hloc[0,:].size                                          # number of harmonics
  yhloc = hloc[0,:]                                            # initialize output frame
  yhmag = hmag[0,:]                                            # initialize output frame
  ystocEnv = stocEnv[0,:]                                      # initialize output frame
  outL = int(max(outTime)/float(max(inTime))*L)                # number of synthesis frames
  outIndexes = (outL-1)*outTime/max(outTime)                   # output indexes
  inIndexes = (L-1)*inTime/max(inTime)                         # input indexes
  interpf = interp1d(outIndexes, inIndexes)                    # generate interpolation function
  il = np.arange(outL)
  indexes = interpf(il)
  for l in indexes[1:]:
    yhloc = np.vstack((yhloc, hloc[round(l),:]))
    yhmag = np.vstack((yhmag, hmag[round(l),:])) 
    ystocEnv = np.vstack((ystocEnv, stocEnv[round(l),:]))
  return yhloc, yhmag, ystocEnv, indexes

def defaultTest():
    str_time = time.time()    
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase.wav'))
    w = np.blackman(801)
    N = 1024
    t = -90
    nH = 50
    minf0 = 350
    maxf0 = 700
    f0et = 5
    maxhd = 0.2
    stocf = 0.2
    hloc, hmag, stocEnv, Ns, H = hpsAnal.hpsAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
    inTime = np.array([0, 0.165, 0.595, 0.850, 1.15, 2.15, 2.81, 3.285, 4.585, 4.845, 5.1, 6.15, 6.825, 7.285, 8.185, 8.830])
    outTime = np.array([0, 0.165, 0.595, 0.850, .9+1.15, .2+2.15, 2.81, 3.285, 4.585, .6+4.845, .4+5.1, 6.15, 6.825, 7.285, 8.185, 8.830])            
    yhloc, yhmag, ystocEnv, indexes = hpsTimeScale(hloc, hmag, stocEnv, inTime, outTime)
    y, yh, yst = hpsSynth.hpsSynth(yhloc, yhmag, ystocEnv, Ns, H, fs)   
    print "time taken for computation " + str(time.time()-str_time)

if __name__ == '__main__':
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase.wav'))
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
  hloc, hmag, stocEnv, Ns, H = hpsAnal.hpsAnal(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
  inTime = np.array([0, 0.165, 0.595, 0.850, 1.15, 2.15, 2.81, 3.285, 4.585, 4.845, 5.1, 6.15, 6.825, 7.285, 8.185, 8.830])
  outTime = np.array([0, 0.165, 0.595, 0.850, .9+1.15, .2+2.15, 2.81, 3.285, 4.585, .6+4.845, .4+5.1, 6.15, 6.825, 7.285, 8.185, 8.830])            
  yhloc, yhmag, ystocEnv, indexes = hpsTimeScale(hloc, hmag, stocEnv, inTime, outTime)
  y, yh, yst = hpsSynth.hpsSynth(yhloc, yhmag, ystocEnv, Ns, H, fs)
  wp.play(y, fs)


