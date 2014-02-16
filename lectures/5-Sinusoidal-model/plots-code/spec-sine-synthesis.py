import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math
from scipy.signal import hamming, blackmanharris

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import smsWavplayer as wp
import dftAnal, dftSynth
M = 255
N = 4096
hM = int(M/2.0)
fs = 44100
f0 = 5000
A0 = .9
ph = 1.5
t = np.arange(-hM,hM+1)/(float fs)
x = A0 * np.cos(2*np.pi*f0*t+ph)
w = hamming(255)
mX, pX = dftAnal.dftAnal(x, w, N)
y = dftSynth.dftSynth(mX,pX,M)*sum(w)
freqaxis = fs*np.arange(0,N/2)/float(N)
taxis = np.arange(N)/float(fs) 

plt.figure(1)
plt.subplot(3,2,1)
plt.plot(freqaxis, mX, 'r')
plt.axis([0,fs/2,-110, 0])
plt.title ('mX; Hamming, f0=1000Hz, A0=.9')

plt.subplot(3,2,3)
plt.plot(freqaxis, pX, 'c')
plt.axis([0,fs/2,min(pX),max(pX)])
plt.title ('pX; Hamming, f0=1000Hz, theta=1.5')

plt.subplot(3,2,5)
plt.plot(np.arange(-hM,hM+1), y[0:M], 'b')
plt.axis([-hM,hM+1,-1,1])
plt.title ('synthesized sine: y')

w = blackmanharris(255)
mX, pX = dftAnal.dftAnal(x, w, N)
y = dftSynth.dftSynth(mX,pX,M)*sum(w)

plt.subplot(3,2,2)
plt.plot(freqaxis, mX, 'r')
plt.axis([0,fs/2,-110, 0])
plt.title ('mX; Blackman Harris, f0=1000Hz, A0=.9')

plt.subplot(3,2,4)
plt.plot(freqaxis, pX, 'c')
plt.axis([0,fs/2,min(pX),max(pX)])
plt.title ('pX; Blackman Harris, f0=1000Hz, theta=1.5')

plt.subplot(3,2,6)
plt.plot(np.arange(-hM,hM+1), y[0:M], 'b')
plt.axis([-hM,hM+1,-1,1])
plt.title ('synthesized sine: y')
plt.show()
