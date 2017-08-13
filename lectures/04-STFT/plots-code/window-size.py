import math
import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftModel as DF
import utilFunctions as UF


(fs, x) = UF.wavread('../../../sounds/oboe-A4.wav')
N = 128
start = int(.81*fs)
x1 = x[start:start+N] 
plt.figure(1, figsize=(9.5, 6))
plt.subplot(321)
plt.plot(np.arange(start, (start+N), 1.0)/fs, x1*np.hamming(N), 'b', lw=1.5)
plt.axis([start/fs, (start+N)/fs, min(x1*np.hamming(N)), max(x1*np.hamming(N))])
plt.title('x1, M = 128')

mX, pX = DF.dftAnal(x1, np.hamming(N), N)
plt.subplot(323)
plt.plot((fs/2.0)*np.arange(mX.size)/float(mX.size), mX, 'r', lw=1.5)
plt.axis([0,fs/2.0,-90,max(mX)])
plt.title('mX1')

plt.subplot(325)
plt.plot((fs/2.0)*np.arange(mX.size)/float(mX.size), pX, 'c', lw=1.5)
plt.axis([0,fs/2.0,min(pX),max(pX)])
plt.title('pX1')

N = 1024
start = int(.81*fs)
x2 = x[start:start+N]
mX, pX = DF.dftAnal(x2, np.hamming(N), N)

plt.subplot(322)
plt.plot(np.arange(start, (start+N), 1.0)/fs, x2*np.hamming(N), 'b', lw=1.5)
plt.axis([start/fs, (start+N)/fs, min(x2), max(x2)])
plt.title('x2, M = 1024')

plt.subplot(324)
plt.plot((fs/2.0)*np.arange(mX.size)/float(mX.size), mX, 'r', lw=1.5)
plt.axis([0,fs/2.0,-90,max(mX)])
plt.title('mX2')

plt.subplot(326)
plt.plot((fs/2.0)*np.arange(mX.size)/float(mX.size), pX, 'c', lw=1.5)
plt.axis([0,fs/2.0,min(pX),max(pX)])
plt.title('pX2')

plt.tight_layout()
plt.savefig('window-size.png')
plt.show()
