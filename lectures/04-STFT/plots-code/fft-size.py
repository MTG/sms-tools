import math
import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftModel as DF
import utilFunctions as UF

(fs, x) = UF.wavread('../../../sounds/oboe-A4.wav')
M = 512
N = 512
start = int(.8*fs)
x1 = x[start:start+M]
xw = x1 * np.hamming(M) 

plt.figure(1, figsize=(9.5, 6))
plt.subplot(311)
plt.plot(np.arange(start, (start+M), 1.0)/fs, xw, 'b', lw=1.5)
plt.axis([start/fs, (start+M)/fs, min(xw), max(xw)])
plt.title('x (oboe-A4.wav), M = 512')
mX, pX = DF.dftAnal(x1, np.hamming(N), N)

plt.subplot(312)
plt.plot((fs/2.0)*np.arange(mX.size)/float(mX.size), mX, 'r', lw=1.5)
plt.axis([0,fs/4.0,-85,max(mX)])
plt.title('mX, N = 512')

M = 512
N = 2048
start = int(.8*fs)
x1 = x[start:start+M]
xw = x1 * np.hamming(M)
mX, pX = DF.dftAnal(x1, np.hamming(M), N)
         
plt.subplot(313)
plt.plot((fs/2.0)*np.arange(mX.size)/float(mX.size), mX, 'r', lw=1.5)
plt.axis([0,fs/4.0,-85,max(mX)])
plt.title('mX, N = 2048')

plt.tight_layout()
plt.savefig('fft-size.png')
plt.show()
