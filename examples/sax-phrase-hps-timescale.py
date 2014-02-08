import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/transformations/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/basicFunctions/'))

import hpsAnal, hpsSynth, hpsTimeScale
import smsWavplayer as wp

if __name__ == '__main__':
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/sax-phrase.wav'))
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
  inTime = np.array([0, 0.165, 0.595, 0.850, 1.15, 2.15, 2.81, 3.285, 4.585, 4.845, 5.1, 6.15, 6.825, 7.285, 8.185, 8.830, 9.379])
  outTime = np.array([0, 0.165, 0.595, 0.850, .9+1.15, 2.15, 2.81, 3.285, 4.585, 4.845, .9+5.1, 6.15, 6.825, 7.285, 8.185, 8.830, 9.379])            
  yhloc, yhmag, ystocEnv, indexes = hpsTimeScale.hpsTimeScale(hloc, hmag, stocEnv, inTime, outTime)
  y, yh, yst = hpsSynth.hpsSynth(yhloc, yhmag, ystocEnv, Ns, H, fs)
  wp.play(y, fs)
    # wp.wavwrite(y,fs,'sax-phrase-total-synthesis.wav')
    # wp.wavwrite(yh,fs,'sax-phrase-harmonic-component.wav')
    # wp.wavwrite(yr,fs,'sax-phrase-residual-component.wav')
    # wp.wavwrite(yst,fs,'sax-phrase-stochastic-component.wav')
