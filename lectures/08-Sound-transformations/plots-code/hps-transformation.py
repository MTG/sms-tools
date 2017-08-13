# function call to the transformation functions of relevance for the hpsModel

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

inputFile='../../../sounds/sax-phrase-short.wav'
window='blackman'
M=601
N=1024
t=-100
minSineDur=0.1
nH=100
minf0=350
maxf0=700
f0et=5
harmDevSlope=0.01
stocf=0.1

Ns = 512
H = 128

(fs, x) = UF.wavread(inputFile)
w = get_window(window, M)
hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
timeScaling = np.array([0, 0, 2.138, 2.138-1.5, 3.146, 3.146])
yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreq, hmag, mYst, timeScaling)

y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)

UF.wavwrite(y,fs, 'hps-transformation.wav')


plt.figure(figsize=(12, 9))

maxplotfreq = 14900.0

# plot the input sound
plt.subplot(4,1,1)
plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.title('x (sax-phrase-short.wav')

# plot spectrogram stochastic compoment
plt.subplot(4,1,2)
numFrames = int(mYst[:,0].size)
sizeEnv = int(mYst[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:int(sizeEnv*maxplotfreq/(.5*fs)+1)]))
plt.autoscale(tight=True)

# plot harmonic on top of stochastic spectrogram
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('harmonics + stochastic residual')


# plot spectrogram of transformed stochastic compoment
plt.subplot(4,1,3)
numFrames = int(ystocEnv[:,0].size)
sizeEnv = int(ystocEnv[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(ystocEnv[:,:int(sizeEnv*maxplotfreq/(.5*fs)+1)]))
plt.autoscale(tight=True)

# plot transformed harmonic on top of stochastic spectrogram
harms = yhfreq*np.less(yhfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('timescaled harmonics + stochastic residual')

# plot the output sound
plt.subplot(4,1,4)
plt.plot(np.arange(y.size)/float(fs), y)
plt.axis([0, y.size/float(fs), min(y), max(y)])
plt.title('output sound: y')

plt.tight_layout()
plt.savefig('hps-transformation.png')
plt.show()
