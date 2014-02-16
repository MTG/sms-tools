import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import smsWavplayer as wp
import dftAnal, dftSynth
M = 255
N = 4096
hM = int(M/2.0)
fs = 44100
f0 = 1000
A0 = .8
ph = 0
x = A0 * np.cos(2*np.pi*(30.5/M)*np.arange(-hM,hM+1)+ph)
w = np.hamming(255)
mX, pX = dftAnal.dftAnal(x, w, N)
y = dftSynth.dftSynth(mX,pX,N)
freqaxis = fs*np.arange(0,N/2)/float(N)
taxis = np.arange(N)/float(fs) 

plt.figure(1)
plt.subplot(3,2,1)
plt.plot(freqaxis, mX, 'r')
plt.axis([0,fs/2,min(mX),max(mX)])
plt.title ('magnitude spectrum: mX')

plt.subplot(3,1,3)
plt.plot(freqaxis, pX, 'c')
plt.axis([0,fs/2,min(pX),max(pX)])
plt.title ('phase spectrum: pX')

plt.subplot(3,1,3)
plt.plot(taxis, y, 'b')
plt.axis([0,N-1,min(pX),20])
plt.title ('phase spectrum: pX')

plt.show()
