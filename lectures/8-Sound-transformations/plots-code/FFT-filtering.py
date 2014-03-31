import math
import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftModel as DFT
import utilFunctions as UF


(fs, x) = UF.wavread('../../../sounds/orchestra.wav')
N = 2048
start = 1.0*fs
x1 = x[start:start+N]

plt.figure(1, figsize=(9.5, 7))
plt.subplot(321)
plt.plot(np.arange(N)/float(fs), x1*np.hamming(N), 'b')
plt.axis([0, N/float(fs), min(x1*np.hamming(N)), max(x1*np.hamming(N))])
plt.title('x (orchestra.wav)')

mX, pX = DFT.dftAnal(x1, np.hamming(N), N)
filter = np.array([0, -40, 200, -40, 300, 0, 600, 0, 700,-40, 1500, -40, 1600, 0, 2500, 0, 2800, -40, 22050, -60])
filt = np.interp(np.arange(N/2), (N/2)*filter[::2]/filter[-2], filter[1::2])
mY = mX + filt

plt.subplot(323)
plt.plot(fs*np.arange(N/2)/float(N/2), mX, 'r', lw=1.3, label = 'mX')
plt.plot(fs*np.arange(N/2)/float(N/2), filt+max(mX), 'k', lw=1.3, label='filter')
plt.legend(prop={'size':10})
plt.axis([0,fs/4.0,-90,max(mX)+2])
plt.title('mX')

plt.subplot(325)
plt.plot(fs*np.arange(N/2)/float(N/2), pX, 'c', lw=1.3)
plt.axis([0,fs/4.0,min(pX),8])
plt.title('pX')

y = DFT.dftSynth(mY, pX, N)*sum(np.hamming(N))
mY1, pY = DFT.dftAnal(y, np.hamming(N), N)
plt.subplot(322)
plt.plot(np.arange(N)/float(fs), y, 'b')
plt.axis([0, float(N)/fs, min(y), max(y)])
plt.title('y')

plt.subplot(324)
plt.plot(fs*np.arange(N/2)/float(N/2), mY1, 'r', lw=1.3)
plt.axis([0,fs/4.0,-90,max(mY1)+2])
plt.title('mY')

plt.subplot(326)
plt.plot(fs*np.arange(N/2)/float(N/2), pY, 'c', lw=1.3)
plt.axis([0,fs/4.0,min(pY),8])
plt.title('pY')

plt.tight_layout()
plt.savefig('FFT-filtering.png')
plt.show()
