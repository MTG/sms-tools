import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import sys, os, functools, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import sineModel as SM
import stft as STFT
import utilFunctions as UF

plt.figure(1, figsize=(9, 6))
plt.subplot(211)
(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/carnatic.wav'))
x1 = x[int(4.35*fs):]
w = np.blackman(1301)
N = 2048
H = 250
t = -70
minSineDur = .02
maxnSines = 150
freqDevOffset = 20
freqDevSlope = 0.02
mX, pX = STFT.stftAnal(x, w, N, H)
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

maxplotfreq = 3000.0
maxplotbin = int(N*maxplotfreq/fs)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(maxplotbin+1)*float(fs)/N 
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))                       
plt.autoscale(tight=True)
  
tracks = tfreq*np.less(tfreq, maxplotfreq)
tracks[tracks<=0] = np.nan
plt.plot(frmTime, tracks, color='k', lw=1.5)
plt.autoscale(tight=True)
plt.title('mX + sine frequencies (carnatic.wav)')

plt.subplot(212)
(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/vignesh.wav'))
w = np.blackman(1101)
N = 2048
H = 250
t = -90
minSineDur = .1
maxnSines = 200
freqDevOffset = 20
freqDevSlope = 0.02
mX, pX = STFT.stftAnal(x, w, N, H)
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

maxplotfreq = 3000.0
maxplotbin = int(N*maxplotfreq/fs)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(maxplotbin+1)*float(fs)/N 
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))                      
plt.autoscale(tight=True)
  
tracks = tfreq*np.less(tfreq, maxplotfreq)
tracks[tracks<=0] = np.nan
plt.plot(frmTime, tracks, color='k', lw=1.5)
plt.autoscale(tight=True)
plt.title('mX + sine frequencies (vignesh.wav)')

plt.tight_layout()
plt.savefig('monophonic-polyphonic.png')
plt.show()
