# example of using the functions in software/models/stft.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import stft as STFT

# ------- analysis parameters -------------------

# input sound file (monophonic with sampling rate of 44100)
inputFile = '../sounds/piano.wav'

# analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris)	
window = 'hamming'

# analysis window size 
M = 1024

# fft size (power of two, bigger or equal than M)
N = 1024  

# hop size (at least 1/2 of analysis window size to have good overlap-add)
H = 512      

# output sound file (monophonic with sampling rate of 44100)
outputFile = 'piano_stft.wav'         

# --------- computation -----------------

# read input sound
(fs, x) = UF.wavread(inputFile)

# compute analysis window
w = get_window(window, M)

# compute the magnitude and phase spectrogram
mX, pX = STFT.stftAnal(x, fs, w, N, H)
 
# perform the inverse stft
y = STFT.stftSynth(mX, pX, M, H)

# write the sound resulting from the inverse stft
UF.wavwrite(y, fs, outputFile)

# --------- plotting --------------------

# create figure to plot
plt.figure(1, figsize=(12, 9))

# frequency range to plot
maxplotfreq = 5000.0

# plot the input sound
plt.subplot(4,1,1)
plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('input sound: x')

# plot magnitude spectrogram
plt.subplot(4,1,2)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N  
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('magnitude spectrogram')
plt.autoscale(tight=True)

# plot the phase spectrogram
plt.subplot(4,1,3)
numFrames = int(pX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
plt.pcolormesh(frmTime, binFreq, np.transpose(np.diff(pX[:,:N*maxplotfreq/fs+1],axis=1)))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('phase spectrogram (derivative)')
plt.autoscale(tight=True)

# plot the output sound
plt.subplot(4,1,4)
plt.plot(np.arange(y.size)/float(fs), y)
plt.axis([0, y.size/float(fs), min(y), max(y)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('output sound: y')

plt.tight_layout()
plt.show()
