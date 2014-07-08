# example of using the functions in software/models/stft.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hann, hamming, blackman
import time, os, sys, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import stft as STFT
import dftModel as DFT

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
(fs, x) = UF.wavread('../sounds/piano.wav')

# analysis window size 
M = 1024

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
w = np.hamming(M) 

# fft size (power of two, bigger or equal than M)
N = 1024  

# hop size (at least 1/2 of analysis window size to have good overlap-add)
H = 512               

# --------- computation -----------------

# compute the magnitude and phase spectrogram
mX, pX = STFT.stftAnal(x, fs, w, N, H)
 
# perform the inverse stft
y = STFT.stftSynth(mX, pX, w.size, H)

# --------- plotting --------------------

# create figure to plot
plt.figure(1, figsize=(9.5, 7))

# plot the magnitude spectrogmra
plt.subplot(211)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('magnitude spectrogram')
plt.autoscale(tight=True)

# plot the phase spectrogram
plt.subplot(212)
numFrames = int(pX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(np.diff(pX,axis=1)))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('phase spectrogram (derivative)')
plt.autoscale(tight=True)

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write the sound resulting from the inverse stft
UF.wavwrite(y, fs, 'piano-stft.wav')   
  

 
