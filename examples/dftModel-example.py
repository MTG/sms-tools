# example of using the functions in software/models/dftModel.py

import matplotlib.pyplot as plt
import numpy as np
import time, os, sys, math
from scipy.fftpack import fft, ifft
from scipy.signal import get_window
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import dftModel as DFT

# ------- analysis parameters -------------------

# input sound file (monophonic with sampling rate of 44100)
inputFile = '../sounds/oboe-A4.wav'

# analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris)	
window = 'blackman'

# analysis window size (odd integer value)
M = 511

# fft size (power of two, bigger than M)
N = 1024  

 # time  to start analysis (in seconds)          
time = .2        

# --------- computation -----------------

# read input sound
(fs, x) = UF.wavread(inputFile)

# compute analysis window
w = get_window(window, M)
	
# get a fragment of the input sound 
x1 = x[int(time*fs):int(time*fs)+M]
 
# compute the dft of the sound fragment
mX, pX = DFT.dftAnal(x1, w, N)

# compute the inverse dft of the spectrum
y = DFT.dftSynth(mX, pX, w.size)*sum(w)

# --------- plotting --------------------

# create figure
plt.figure(1, figsize=(12, 9))

# plot the sound fragment
plt.subplot(4,1,1)
plt.plot(time + np.arange(M)/float(fs), x1)
plt.axis([time, time + M/float(fs), min(x1), max(x1)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('input sound: x')

# plot the magnitude spectrum
plt.subplot(4,1,2)
plt.plot((fs/2.0)*np.arange(N/2)/float(N/2), mX, 'r')
plt.axis([0, fs/2.0, min(mX), max(mX)])
plt.title ('magnitude spectrum: mX')
plt.ylabel('amplitude (dB)')
plt.xlabel('frequency (Hz)')

# plot the phase spectrum
plt.subplot(4,1,3)
plt.plot((fs/2.0)*np.arange(N/2)/float(N/2), pX, 'c')
plt.axis([0, fs/2.0, min(pX), max(pX)])
plt.title ('phase spectrum: pX')
plt.ylabel('phase (radians)')
plt.xlabel('frequency (Hz)')

# plot the sound resulting from the inverse dft
plt.subplot(4,1,4)
plt.plot(time + np.arange(M)/float(fs), y)
plt.axis([time, time + M/float(fs), min(y), max(y)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('output sound: y')

plt.tight_layout()
plt.show()
