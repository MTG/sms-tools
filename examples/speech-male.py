import numpy as np
import matplotlib.pyplot as plt
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))

import hprModel as HPR
import utilFunctions as UF

if __name__ == '__main__':
    (fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/speech-male.wav'))
    w = np.blackman(1001)
    N = 1024
    t = -90
    nH = 100
    minf0 = 60
    maxf0 = 180
    f0et = 3
    y, yh, xr = HPR.hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et)
     
    # UF.play(y, fs)
    # UF.play(yh, fs)
    # UF.play(yr, fs)
    # UF.play(yst, fs)
    UF.wavwrite(y,fs,'speech-male-synthesis.wav')
    UF.wavwrite(yh,fs,'speech-male-harmonic-component.wav')
    UF.wavwrite(xr,fs,'speech-male-residual-component.wav')
