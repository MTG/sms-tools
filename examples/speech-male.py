import numpy as np
import matplotlib.pyplot as plt
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/basicFunctions/'))

import hprModel, hpsModel
import smsWavplayer as wp

if __name__ == '__main__':
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/speech-male.wav'))
    w = np.blackman(1001)
    N = 1024
    t = -90
    nH = 100
    minf0 = 60
    maxf0 = 180
    f0et = 3
    maxhd = 0.2
    maxFreq = 1000
    start = 0*fs
    end = x.size
    maxnpeaksTwm = 5
    stocf = .2
    y, yh, yr = hprModel.hprModel(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm)
     
    # wp.play(y, fs)
    # wp.play(yh, fs)
    # wp.play(yr, fs)
    # wp.play(yst, fs)
    wp.wavwrite(y,fs,'speech-male-synthesis.wav')
    wp.wavwrite(yh,fs,'speech-male-harmonic-component.wav')
    wp.wavwrite(yr,fs,'speech-male-residual-component.wav')
