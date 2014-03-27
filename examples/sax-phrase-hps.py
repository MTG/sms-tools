import numpy as np
import matplotlib.pyplot as plt
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))

import hpsModel as HPS
import utilFunctions as UF

if __name__ == '__main__':
    (fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/sax-phrase.wav'))
    w = np.blackman(601)
    N = 1024
    t = -100
    nH = 100
    minf0 = 350
    maxf0 = 700
    f0et = 5
    maxnpeaksTwm = 5
    minSineDur = .1
    harmDevSlope = 0.01
    Ns = 512
    H = Ns/4
    stocf = .2
    hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
    y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, mYst, Ns, H, fs)

    # UF.play(y, fs)
    # UF.play(yh, fs)
    # UF.play(yst, fs)
    UF.wavwrite(y,fs,'sax-phrase-total-synthesis.wav')
    UF.wavwrite(yh,fs,'sax-phrase-harmonic-component.wav')
    UF.wavwrite(yst,fs,'sax-phrase-stochastic-component.wav')
