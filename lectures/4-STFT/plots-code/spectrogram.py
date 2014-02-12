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
w = np.hamming(1001)
N = 1024
H = 256
mX, pX = stftAnal.stftAnal(x, fs, w, N, H)

plt.figure(1)
plt.subplot(211)
numFrames = int(mX[1,:].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, mX)
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('piano.wav magnitude spectrogram; M=1001, N=1024, H=256')
plt.autoscale(tight=True)

plt.subplot(212)
numFrames = int(pX[1,:].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.diff(pX,axis=0))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('piano.wav phase spectrogram (derivative); M=1001, N=1024, H=256')
plt.autoscale(tight=True)
plt.show()
