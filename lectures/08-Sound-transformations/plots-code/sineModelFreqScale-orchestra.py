import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/transformations/'))

import sineModel as SM 
import stft as STFT
import utilFunctions as UF
import sineTransformations as SMT

(fs, x) = UF.wavread('../../../sounds/orchestra.wav')
w = np.hamming(801)
N = 2048
t = -90
minSineDur = .005
maxnSines = 150
freqDevOffset = 20
freqDevSlope = 0.02
Ns = 512
H = Ns//4
mX, pX = STFT.stftAnal(x, w, N, H)
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
freqScaling = np.array([0, .8, 1, 1.2])           
ytfreq = SMT.sineFreqScaling(tfreq, freqScaling)
y = SM.sineModelSynth(ytfreq, tmag, np.array([]), Ns, H, fs)
mY, pY = STFT.stftAnal(y, w, N, H)
UF.wavwrite(y,fs, 'sineModelFreqScale-orchestra.wav')

maxplotfreq = 4000.0

plt.figure(1, figsize=(9.5, 7))
plt.subplot(4,1,1)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.axis([0,x.size/float(fs),min(x),max(x)])
plt.title('x (orchestra.wav)')                        

plt.subplot(4,1,2)
numFrames = int(tfreq[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)
tracks = tfreq*np.less(tfreq, maxplotfreq)
tracks[tracks<=0] = np.nan
plt.plot(frmTime, tracks, color='k', lw=1)
plt.autoscale(tight=True)
plt.title('sine frequencies')  

maxplotbin = int(N*maxplotfreq/fs)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
plt.autoscale(tight=True)

plt.subplot(4,1,3)
numFrames = int(ytfreq[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)
tracks = ytfreq*np.less(ytfreq, maxplotfreq)
tracks[tracks<=0] = np.nan
plt.plot(frmTime, tracks, color='k', lw=1)
plt.autoscale(tight=True)
plt.title('freq-scaled sine frequencies') 

maxplotbin = int(N*maxplotfreq/fs)
numFrames = int(mY[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:,:maxplotbin+1]))
plt.autoscale(tight=True) 

plt.subplot(4,1,4)
plt.plot(np.arange(y.size)/float(fs), y, 'b')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('y')    

plt.tight_layout()
plt.savefig('sineModelFreqScale-orchestra.png')
plt.show()

