import numpy as np
import matplotlib.pyplot as plt
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))

import hprModel as HPR
import hpsModel as HPS
import utilFunctions as UF

if __name__ == '__main__':
    (fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/speech-female.wav'))
    w = np.blackman(901)
    N = 1024
    t = -80
    nH = 30
    minf0 = 100
    maxf0 = 400
    f0et = 2
    maxFreq = 800
    start = 0*fs
    end = x.size
    stocf = .2
    y, yh, yr = HPR.hprModel(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et)
    y, yh, yst = HPS.hpsModel(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et, stocf)
 
    # wp.play(y, fs)
    # wp.play(yh, fs)
    # wp.play(yr, fs)
    # wp.play(yst, fs)
    UF.wavwrite(y,fs,'speech-female-synthesis.wav')
    UF.wavwrite(yh,fs,'speech-female-harmonic-component.wav')
    UF.wavwrite(yr,fs,'speech-female-residual-component.wav')
    UF.wavwrite(yst,fs,'speech-female-stochastic-component.wav')
