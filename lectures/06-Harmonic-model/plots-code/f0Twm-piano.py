import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackman
import math
import sys, os, functools, time
from smstools.models import dftModel as DFT
from smstools.models import utilFunctions as UF
from smstools.models import stft as STFT
from smstools.models import sineModel as SM
from smstools.models import harmonicModel as HM

(fs, x) = UF.wavread('../../../sounds/piano.wav')
w = np.blackman(1501)
N = 2048
t = -90
minf0 = 100
maxf0 = 300
f0et = 1
maxnpeaksTwm = 4
H = 128
x1 = x[int(1.5*fs):int(1.8*fs)]

plt.figure(1, figsize=(9, 7))
mX, pX = STFT.stftAnal(x, w, N, H)
f0 = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)
f0 = UF.cleaningTrack(f0, 5)
yf0 = UF.sinewaveSynth(f0, .8, H, fs)
f0[f0==0] = np.nan
maxplotfreq = 800.0
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = fs*np.arange(N*maxplotfreq/fs)/N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:int(N*maxplotfreq/fs+1)]))
plt.autoscale(tight=True)

plt.plot(frmTime, f0, linewidth=2, color='k')
plt.autoscale(tight=True)
plt.title('mX + f0 (piano.wav), TWM')

plt.tight_layout()
plt.savefig('f0Twm-piano.png')
UF.wavwrite(yf0, fs, 'f0Twm-piano.wav')
plt.show()

