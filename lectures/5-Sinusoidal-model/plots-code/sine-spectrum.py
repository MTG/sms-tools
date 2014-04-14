import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftModel as DFT
M = 151
N = 1024
hM = int(M/2.0)
fs = 44100
f0 = 2000.0
A0 = .8
ph = 0
x = A0 * np.cos(2*np.pi*(f0/fs)*np.arange(-hM,hM+1)+ph)
w = np.hamming(M)
mX, pX = DFT.dftAnal(x, w, N)
freqaxis = fs*np.arange(0,N/2)/float(N)
taxis = np.arange(N)/float(fs) 

plt.figure(1, figsize=(9.5, 7))

plt.subplot(3,1,1)
plt.plot(np.arange(M)/float(fs), x, 'b', lw=1.5)
plt.axis([0,(M-1)/float(fs),min(x),max(x)])
plt.title ('x')

plt.subplot(3,1,2)
plt.plot(freqaxis, mX, 'r', lw=1.5)
plt.axis([0,fs/6,-80,max(mX)])
plt.title ('mX')

plt.subplot(3,1,3)
plt.plot(freqaxis, pX, 'c', lw=1.5)
plt.axis([0,fs/6,-8,max(pX)+1])
plt.title ('pX')


plt.tight_layout()
plt.savefig('sine-spectrum.png')
plt.show()
