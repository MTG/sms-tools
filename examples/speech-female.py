import numpy as np
import matplotlib.pyplot as plt
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/basicFunctions/'))

import hprModel, hpsModel, hprModelSpectrogramPlot
import smsWavplayer as wp

if __name__ == '__main__':
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/speech-female.wav'))
    w = np.blackman(901)
    N = 1024
    t = -80
    nH = 30
    minf0 = 100
    maxf0 = 400
    f0et = 2
    maxhd = 0.1
    maxFreq = 800
    start = 0*fs
    end = x.size
    maxnpeaksTwm = 5
    stocf = .2
    y, yh, yr = hprModel.hprModel(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm)
    y, yh, yst = hpsModel.hpsModel(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
 
    # wp.play(y, fs)
    # wp.play(yh, fs)
    # wp.play(yr, fs)
    # wp.play(yst, fs)
    wp.wavwrite(y,fs,'speech-female-synthesis.wav')
    wp.wavwrite(yh,fs,'speech-female-harmonic-component.wav')
    wp.wavwrite(yr,fs,'speech-female-residual-component.wav')
    wp.wavwrite(yst,fs,'speech-female-stochastic-component.wav')
