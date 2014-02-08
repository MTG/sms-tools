import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))

import smsWavplayer as wp
import hprModel as HPR

if __name__ == '__main__':
  (fs, x) = wp.wavread('cello-G2.wav')
  w = np.blackman(2301)
  N = 4096
  t = -90
  nH = 100
  minf0 = 80
  maxf0 = 110
  f0et = 10
  maxhd = 0.2
  maxnpeaksTwm = 5
  y, yh, yr = HPR.hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm)
  wp.play(y, fs)
  wp.play(yh, fs)
  wp.play(yr, fs)
