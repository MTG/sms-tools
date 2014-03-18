import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/utilFunctions/'))

import stftAnal as STFT
import waveIO as WIO
import harmonicModelAnal as HA
import harmonicModelSynth as HS
import sineSubtraction as SS


(fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/cello-double.wav'))
w = np.blackman(3501)
N = 2048*2
t = -100
nH = 100
minf0 = 140
maxf0 = 150
f0et = 10
maxnpeaksTwm = 5
minSineDur = .2
harmDevSlope = 0.001
Ns = 512
H = Ns/4

hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
y = HS.harmonicModelSynth(hfreq, hmag, hphase, Ns, H, fs)
xr = SS.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)

WIO.play(x, fs)
WIO.play(y, fs)
WIO.play(xr, fs)
