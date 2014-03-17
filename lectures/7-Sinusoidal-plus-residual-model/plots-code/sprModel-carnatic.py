import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import ifft, fftshift
import math

import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import sineModelAnal as SA
import sineModelSynth as SS
import waveIO as WIO
import sineSubtraction as SSub


(fs, x) = WIO.wavread('../../../sounds/carnatic.wav')
w = np.blackman(2001)
N = 2048
t = -90
minSineDur = .2
maxnSines = 200
freqDevOffset = 20
freqDevSlope = 0.02
Ns = 512
H = Ns/4
tfreq, tmag, tphase = SA.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
y = SS.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

xr = SSub.sineSubtraction(x, Ns, H, tfreq, tmag, tphase, fs)
#mXr, pXr = ST.stftAnal(xr, fs, hamming(H*2), H*2, H)

WIO.play(xr, fs)
