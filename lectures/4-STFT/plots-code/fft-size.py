import math
import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftAnal, dftSynth
import smsWavplayer as wp
from scipy.io.wavfile import read

(fs, x) = wp.wavread('../../../sounds/oboe-A4.wav')
M = 512
N = 512
start = .8*fs
x1 = x[start:start+M]
xw = x1 * np.hamming(M) 
plt.figure(1)
plt.subplot(311)
plt.plot(np.arange(start, (start+M), 1.0)/fs, xw, 'b')
plt.axis([start/fs, (start+M)/fs, min(xw), max(xw)])
plt.xlabel('time (sec)')
plt.title('x = wavread(oboe-A4.wav), M = 512')
mX, pX = dftAnal.dftAnal(x1, np.hamming(N), N)

plt.subplot(312)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX, 'r')
plt.axis([0,fs/4.0,-85,max(mX)])
plt.xlabel('frequency (Hz)')
plt.title('abs(X1), N = 512')

M = 512
N = 2048
start = .8*fs
x1 = x[start:start+M]
xw = x1 * np.hamming(M)
mX, pX = dftAnal.dftAnal(x1, np.hamming(M), N)
         
plt.subplot(313)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX, 'r')
plt.axis([0,fs/4.0,-85,max(mX)])
plt.xlabel('frequency (Hz)')
plt.title('abs(X2), N = 2048')

plt.show()
