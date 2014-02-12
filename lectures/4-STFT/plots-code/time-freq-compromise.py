import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))

import stftAnal
import smsWavplayer as wp
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft
import math

(fs, x) = wp.wavread('../../../sounds/piano.wav')



plt.figure(1)

w = np.hamming(256)
N = 256
H = 128
mX1, pX1 = stftAnal.stftAnal(x, fs, w, N, H)
plt.subplot(211)
numFrames = int(mX1[1,:].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, mX1)
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('piano.wav magnitude spectrogram; M=256, N=256, H=128')
plt.autoscale(tight=True)

w = np.hamming(1024)
N = 1024
H = 128
mX2, pX2 = stftAnal.stftAnal(x, fs, w, N, H)

plt.subplot(212)
numFrames = int(mX2[1,:].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, mX2)
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('piano.wav magnitude spectrogram; M=1024, N=1024, H=128')
plt.autoscale(tight=True)
plt.show()

