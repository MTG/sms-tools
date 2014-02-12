import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/basicFunctions/'))

import smsWavplayer as wp

(fs, x) = wp.wavread('../../../sounds/soprano-E4.wav')
N = 1024
x1 = np.blackman(N)*x[40000:40000+N]

plt.figure(1)
X = np.array([])
x2 = np.zeros(N)

plt.subplot(4,1,1)
plt.title ('x = wavread(soprano-E4.wav)')
plt.plot(x1)
plt.axis([0,N,min(x1),max(x1)])

X = fft(fftshift(x1))
mX = 20*np.log10(np.abs(X[0:N/2]))
pX = np.angle(X[0:N/2])

plt.subplot(4,1,2)
plt.title ('mX = magnitude spectrum')
plt.plot(mX, 'r')
plt.axis([0,N/2,min(mX), max(mX)])

plt.subplot(4,1,3)
plt.title ('pX1 = phase spectrum')
plt.plot(pX, 'c')
plt.axis([0,N/2,min(pX),max(pX)])

pX1 = np.unwrap(pX)
plt.subplot(4,1,4)
plt.title ('pX2: unwrapped phase spectrum')
plt.plot(pX1, 'c')
plt.axis([0,N/2,min(pX1),max(pX1)])

plt.show()
