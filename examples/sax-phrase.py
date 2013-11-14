import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../code/spectralModels/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../code/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../code/spectralModelsPlotting/'))

import hprModel, hpsModel, hprModelSpectrogramPlot
import smsWavplayer as wp

if __name__ == '__main__':
    
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/sax-phrase.wav'))
    w = np.blackman(801)
    N = 1024
    t = -90
    nH = 50
    minf0 = 340
    maxf0 = 700
    f0et = 7
    maxhd = 0.2
    maxFreq = 2000
    start = 0*fs
    end = x.size
    maxnpeaksTwm = 5
    stocf = .2
    y, yh, yr = hprModel.hprModel(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxnpeaksTwm)
    y, yh, yst = hpsModel.hpsModel(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, maxnpeaksTwm)
    # hprModelSpectrogramPlot.hprModelPlot(x[start:end], fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, maxFreq, maxnpeaksTwm)

    # wp.play(y, fs)
    # wp.play(yh, fs)
    # wp.play(yr, fs)
    # wp.play(yst, fs)
    wp.wavwrite(y,fs,'sax-phrase-total-synthesis.wav')
    wp.wavwrite(yh,fs,'sax-phrase-harmonic-component.wav')
    wp.wavwrite(yr,fs,'sax-phrase-residual-component.wav')
    wp.wavwrite(yst,fs,'sax-phrase-stochastic-component.wav')