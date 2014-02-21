import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import waveIO as WIO
import stftAnal, sineModelAnal

plt.figure(1)
plt.subplot(211)
(fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/carnatic.wav'))
w = np.blackman(251)
N = 1024
H = 1000
t = -90
mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
ploc, pmag, pphase = sineModelAnal.sineModelAnal(x, fs, w, N, H, t)

maxplotbin = int(N*5000.0/fs)
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
plt.title('spectral peaks on spectrogram (vibraphone-C6.wav)')

plt.subplot(212)
(fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/vignesh.wav'))
w = np.blackman(801)
N = 1024
H = 700
t = -70
mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
ploc, pmag, pphase = sineModelAnal.sineModelAnal(x, fs, w, N, H, t)

maxplotbin = int(N*5000.0/fs)
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
plt.title('spectral peaks on spectrogram (oboe-A4.wav)')

plt.show()
