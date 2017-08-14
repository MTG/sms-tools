import numpy as np
import matplotlib.pyplot as plt
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import stft as STFT
import sineModel as SM
import utilFunctions as UF

(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/flute-A4.wav'))
w = np.blackman(601)
N = 1024
H = 150
t = -80
minSineDur = .1
maxnSines = 150
mX, pX = STFT.stftAnal(x, w, N, H)
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur)

plt.figure(1, figsize=(9.5, 5))
maxplotfreq = 5000.0
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
plt.title('mX + sinusoidal tracks (flute-A4.wav)')

plt.tight_layout()
plt.savefig('sineModelAnal-flute.png')
plt.show()
