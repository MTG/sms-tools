import numpy as np
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models'))

import hprModel as HPR
import smsWavplayer as wp

if __name__ == '__main__':
  (fs, x) = wp.wavread('organ.wav'))
  w = np.blackman(901)
  N = 2048
  t = -100
  nH = 200
  minf0 = 160
  maxf0 = 240
  f0et = 10
  maxhd = 0.1
  y, yh, yr1 = hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd)
  y, yh, yr2 = hprModel(yh, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd)
  y, yh, yr3 = hprModel(yh, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd)
  y, yh, yr4 = hprModel(yh, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd)

  wp.play(yr1, fs)
  wp.play(yr4, fs)
