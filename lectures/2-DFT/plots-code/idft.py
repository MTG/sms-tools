import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('../../software/basicFunctions/')
sys.path.append('../../software/models/')

import dftAnal, dftSynth
import math

k0 = 8.5
N = 64
w = np.ones(N)
x = np.cos(2*np.pi*k0/N*np.arange(-N/2,N/2))
mX, pX = dftAnal.dftAnal(x, w, N)
y = dftSynth.dftSynth(mX, pX, N)

plt.figure(1, figsize=(9.5, 5))
plt.subplot(311)
plt.title('positive freq. magnitude spectrum in dB: mX')
plt.plot(np.arange(0, N/2), mX, 'r', lw=1.5)
plt.axis([0,N/2,min(mX),max(mX)+1])

plt.subplot(312)
plt.title('positive freq. phase spectrum: pX')
plt.plot(np.arange(0, N/2), pX, 'c', lw=1.5)
plt.axis([0,N/2,-np.pi,np.pi])

plt.subplot(313)
plt.title('inverse spectrum: IDFT(X)')
plt.plot(np.arange(-N/2, N/2), y,'b', lw=1.5)
plt.axis([-N/2,N/2-1,min(y), max(y)])

plt.tight_layout()
plt.show()
