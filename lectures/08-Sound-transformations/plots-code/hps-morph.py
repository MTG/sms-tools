# function for doing a morph between two sounds using the hpsModel

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/transformations/'))
import hpsModel as HPS
import hpsTransformations as HPST
import harmonicTransformations as HT
import utilFunctions as UF

inputFile1='../../../sounds/violin-B3.wav'
window1='blackman'
M1=1001
N1=1024
t1=-100
minSineDur1=0.05
nH=60
minf01=200
maxf01=300
f0et1=10
harmDevSlope1=0.01
stocf=0.1

inputFile2='../../../sounds/soprano-E4.wav'
window2='blackman'
M2=901
N2=1024
t2=-100
minSineDur2=0.05
minf02=250
maxf02=500
f0et2=10
harmDevSlope2=0.01

Ns = 512
H = 128

(fs1, x1) = UF.wavread(inputFile1)
(fs2, x2) = UF.wavread(inputFile2)
w1 = get_window(window1, M1)
w2 = get_window(window2, M2)
hfreq1, hmag1, hphase1, stocEnv1 = HPS.hpsModelAnal(x1, fs1, w1, N1, H, t1, nH, minf01, maxf01, f0et1, harmDevSlope1, minSineDur1, Ns, stocf)
hfreq2, hmag2, hphase2, stocEnv2 = HPS.hpsModelAnal(x2, fs2, w2, N2, H, t2, nH, minf02, maxf02, f0et2, harmDevSlope2, minSineDur2, Ns, stocf)

hfreqIntp = np.array([0, .5, 1, .5])
hmagIntp = np.array([0, .5, 1, .5])
stocIntp = np.array([0, .5, 1, .5])
yhfreq, yhmag, ystocEnv = HPST.hpsMorph(hfreq1, hmag1, stocEnv1, hfreq2, hmag2, stocEnv2, hfreqIntp, hmagIntp, stocIntp)

y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs1)

UF.wavwrite(y,fs1, 'hps-morph.wav')

plt.figure(figsize=(12, 9))
frame = 200

plt.subplot(2,3,1)
plt.vlines(hfreq1[frame,:], -100, hmag1[frame,:], lw=1.5, color='b')
plt.axis([0,5000, -80, -15])
plt.title('x1: harmonics')

plt.subplot(2,3,2)
plt.vlines(hfreq2[frame,:], -100, hmag2[frame,:], lw=1.5, color='r')
plt.axis([0,5000, -80, -15])
plt.title('x2: harmonics')

plt.subplot(2,3,3)
yhfreq[frame,:][yhfreq[frame,:]==0] = np.nan
plt.vlines(yhfreq[frame,:], -100, yhmag[frame,:], lw=1.5, color='c')
plt.axis([0,5000, -80, -15])
plt.title('y: harmonics')

stocaxis = (fs1/2)*np.arange(stocEnv1[0,:].size)/float(stocEnv1[0,:].size)
plt.subplot(2,3,4)
plt.plot(stocaxis, stocEnv1[frame,:], lw=1.5, marker='x', color='b')
plt.axis([0,20000, -73, -27])
plt.title('x1: stochastic')

plt.subplot(2,3,5)
plt.plot(stocaxis, stocEnv2[frame,:], lw=1.5, marker='x', color='r')
plt.axis([0,20000, -73, -27])
plt.title('x2: stochastic')

plt.subplot(2,3,6)
plt.plot(stocaxis, ystocEnv[frame,:], lw=1.5, marker='x', color='c')
plt.axis([0,20000, -73, -27])
plt.title('y: stochastic')

plt.tight_layout()
plt.savefig('hps-morph.png')
plt.show()
