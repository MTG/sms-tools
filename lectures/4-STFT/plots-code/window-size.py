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
N = 128
start = .81*fs
x1 = x[start:start+N] 
plt.figure(1)
plt.subplot(321)
plt.plot(np.arange(start, (start+N), 1.0)/fs, x1*np.hamming(N), 'b')
plt.axis([start/fs, (start+N)/fs, min(x1*np.hamming(N)), max(x1*np.hamming(N))])
plt.title('x1, M = 128')
plt.xlabel('time (sec)')

mX, pX = dftAnal.dftAnal(x1, np.hamming(N), N)
plt.subplot(323)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX, 'r')
plt.axis([0,fs/2.0,-90,max(mX)])
plt.title('mX1')
plt.xlabel('frequency (Hz)')

plt.subplot(325)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), pX, 'c')
plt.axis([0,fs/2.0,min(pX),max(pX)])
plt.title('pX1')
plt.xlabel('frequency (Hz)')

N = 1024
start = .81*fs
x2 = x[start:start+N]
mX, pX = dftAnal.dftAnal(x2, np.hamming(N), N)

plt.subplot(322)
plt.plot(np.arange(start, (start+N), 1.0)/fs, x2*np.hamming(N), 'b')
plt.axis([start/fs, (start+N)/fs, min(x2), max(x2)])
plt.title('x2, M = 1024')
plt.xlabel('time (sec)')

plt.subplot(324)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX, 'r')
plt.axis([0,fs/2.0,-90,max(mX)])
plt.title('mX2')
plt.xlabel('frequency (Hz)')

plt.subplot(326)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), pX, 'c')
plt.axis([0,fs/2.0,min(pX),max(pX)])
plt.title('pX2')
plt.xlabel('frequency (Hz)')
plt.show()
