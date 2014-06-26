import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
import time, os, sys
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import stft as STFT
import dftModel as DFT

# read the sound of the piano
(fs, x) = UF.wavread('../sounds/piano.wav')
	
w = np.hamming(1024)  # compute a window of the same size than the FFT
N = 1024              # fft size 
H = 512               # hop size 1/4 of window size to have good overlap-add

# compute the magnitude and phase spectrogram
mX, pX = STFT.stftAnal(x, fs, w, N, H)
  
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

# perform the inverse stft
y = STFT.stftSynth(mX, pX, w.size, H)

# write the sound resulting from the inverse stft
UF.wavwrite(y, fs, 'piano-stft.wav')   
  
plt.tight_layout()
plt.show()
 
