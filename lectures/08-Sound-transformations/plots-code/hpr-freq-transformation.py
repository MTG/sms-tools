# function call to the transformation functions of relevance for the hpsModel

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/transformations/'))
import hprModel as HPR
import stft as STFT
import harmonicTransformations as HT
import utilFunctions as UF

inputFile='../../../sounds/flute-A4.wav'
window='blackman'
M=801
N=2048
t=-90 
minSineDur=0.1
nH=40
minf0=350
maxf0=700
f0et=8
harmDevSlope=0.1
Ns = 512
H = 128

(fs, x) = UF.wavread(inputFile)
w = get_window(window, M)
hfreq, hmag, hphase, xr = HPR.hprModelAnal(x, fs, w, N, H, t, minSineDur, nH, minf0, maxf0, f0et, harmDevSlope)

mXr, pXr = STFT.stftAnal(xr, w, N, H)

freqScaling = np.array([0, 1.5, 1, 1.5])
freqStretching = np.array([0, 1.1, 1, 1.1])
timbrePreservation = 1

hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

y, yh = HPR.hprModelSynth(hfreqt, hmagt, np.array([]), xr, Ns, H, fs)

UF.wavwrite(y,fs, 'hpr-freq-transformation.wav')

plt.figure(figsize=(12, 9))

maxplotfreq = 15000.0

plt.subplot(4,1,1)
plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.title('x (flute-A4.wav)')

plt.subplot(4,1,2)
maxplotbin = int(N*maxplotfreq/fs)
numFrames = int(mXr[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                       
binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:,:maxplotbin+1]))
plt.autoscale(tight=True)

harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('harmonics + residual spectrogram')

plt.subplot(4,1,3)
maxplotbin = int(N*maxplotfreq/fs)
numFrames = int(mXr[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                       
binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:,:maxplotbin+1]))
plt.autoscale(tight=True)

harms = hfreqt*np.less(hfreqt,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.autoscale(tight=True)
plt.title('transposed and stretched harmonics + residual spectrogram')

plt.subplot(4,1,4)
plt.plot(np.arange(y.size)/float(fs), y)
plt.axis([0, y.size/float(fs), min(y), max(y)])
plt.title('y')

plt.tight_layout()
plt.savefig('hpr-freq-transformations.png')
plt.show()
