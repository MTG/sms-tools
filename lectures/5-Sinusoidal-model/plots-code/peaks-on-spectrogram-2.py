import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import stftAnal, sineModelAnal
import smsWavplayer as wp


(fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/flute-A4.wav'))
w = np.hamming(601)
N = 1024
H = 1000
t = -60
mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
ploc, pmag, pphase = sineModelAnal.sineModelAnal(x, fs, w, N, H, t)

maxplotbin = int(N*3000.0/fs)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
plt.autoscale(tight=True)
  
peaks = ploc*np.less(ploc,maxplotbin)*float(fs)/N
peaks[peaks==0] = np.nan
numFrames = int(ploc[:,0].size)
plt.plot(frmTime, peaks, 'x', color='k')
plt.autoscale(tight=True)
plt.title('spectral peaks on spectrogram (flute-A4.wav)')
plt.show()
