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
nH=40
minf01=200
maxf01=300
f0et1=10
harmDevSlope1=0.01
stocf=0.2

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

hfreqIntp = np.array([0, 0, .1, 0, .9, 1, 1, 1])
hmagIntp = np.array([0, 0, .1, 0, .9, 1, 1, 1])
stocIntp = np.array([0, 0, .1, 0, .9, 1, 1, 1])
yhfreq, yhmag, ystocEnv = HPST.hpsMorph(hfreq1, hmag1, stocEnv1, hfreq2, hmag2, stocEnv2, hfreqIntp, hmagIntp, stocIntp)

y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs1)

UF.wavwrite(y,fs1, 'hps-morph-total.wav')

plt.figure(figsize=(12, 9))

# frequency range to plot
maxplotfreq = 15000.0

# plot spectrogram stochastic component of sound 1
plt.subplot(3,1,1)
numFrames = int(stocEnv1[:,0].size)
sizeEnv = int(stocEnv1[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs1)
binFreq = (.5*fs1)*np.arange(sizeEnv*maxplotfreq/(.5*fs1))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv1[:,:int(sizeEnv*maxplotfreq/(.5*fs1)+1)]))
plt.autoscale(tight=True)

# plot harmonic on top of stochastic spectrogram of sound 1
harms = hfreq1*np.less(hfreq1,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs1) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('x1 (violin-B3.wav): harmonics + stochastic spectrogram')

# plot spectrogram stochastic component of sound 2
plt.subplot(3,1,2)
numFrames = int(stocEnv2[:,0].size)
sizeEnv = int(stocEnv2[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs2)
binFreq = (.5*fs2)*np.arange(sizeEnv*maxplotfreq/(.5*fs2))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(stocEnv2[:,:int(sizeEnv*maxplotfreq/(.5*fs2)+1)]))
plt.autoscale(tight=True)

# plot harmonic on top of stochastic spectrogram of sound 2
harms = hfreq2*np.less(hfreq2,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs2) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('x2 (soprano-E4.wav): harmonics + stochastic spectrogram')

# plot spectrogram of transformed stochastic compoment
plt.subplot(3,1,3)
numFrames = int(ystocEnv[:,0].size)
sizeEnv = int(ystocEnv[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs1)
binFreq = (.5*fs1)*np.arange(sizeEnv*maxplotfreq/(.5*fs1))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(ystocEnv[:,:int(sizeEnv*maxplotfreq/(.5*fs1)+1)]))
plt.autoscale(tight=True)

# plot transformed harmonic on top of stochastic spectrogram
harms = yhfreq*np.less(yhfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs1) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('y: harmonics + stochastic spectrogram')

plt.tight_layout()
plt.savefig('hps-morph-total.png')
plt.show()
	

