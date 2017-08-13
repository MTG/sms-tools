import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF
import hpsModel as HPS


(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/sax-phrase-short.wav'))
w = np.blackman(601)
N = 1024
t = -100
nH = 100
minf0 = 350
maxf0 = 700
f0et = 5
minSineDur = .1
harmDevSlope = 0.01
Ns = 512
H = Ns//4
stocf = .2
hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, mYst, Ns, H, fs)

maxplotfreq = 10000.0
plt.figure(1, figsize=(9, 7))

plt.subplot(311)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.autoscale(tight=True)
plt.title('x (sax-phrase-short.wav)')

plt.subplot(312)
numFrames = int(mYst[:,0].size)
sizeEnv = int(mYst[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:int(sizeEnv*maxplotfreq/(.5*fs)+1)]))

harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('harmonics + stochastic')

plt.subplot(313)
plt.plot(np.arange(y.size)/float(fs), y, 'b')
plt.autoscale(tight=True)
plt.title('y')

plt.tight_layout()
plt.savefig('hpsModel-sax-phrase.png')
UF.wavwrite(y, fs, 'sax-phrase-hps-synthesis.wav')
UF.wavwrite(yh, fs, 'sax-phrase-harmonic.wav')
UF.wavwrite(yst, fs, 'sax-phrase-stochastic.wav')
plt.show()
