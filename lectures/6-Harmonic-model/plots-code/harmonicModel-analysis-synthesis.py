import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import harmonicModelAnal as HA
import harmonicModelSynth as HS
import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH


(fs, x) = WIO.wavread('../../../sounds/vignesh.wav')
w = np.blackman(1201)
N = 2048
t = -90
nH = 100
minf0 = 130
maxf0 = 300
f0et = 7
maxnpeaksTwm = 4
Ns = 512
H = Ns/4
minSineDur = .1
harmDevSlope = 0.01

hfreq, hmag, hphase = HA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, maxnpeaksTwm, minSineDur)
y = HS.harmonicModelSynth(hfreq, hmag, hphase, fs)

numFrames = int(hfreq[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)

plt.figure(1)

plt.subplot(3,1,1)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.axis([0,x.size/float(fs),min(x),max(x)])
plt.title('x (vignesh.wav)')                        

plt.subplot(3,1,2)
yhfreq = hfreq
yhfreq[hfreq==0] = np.nan
plt.plot(frmTime, hfreq, color='k')
plt.axis([0,y.size/float(fs),0,8000])
plt.title('harmonic frequencies')

plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y, 'b')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('yh')    

plt.show()

