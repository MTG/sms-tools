import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
from scipy.fftpack import fft, ifft, fftshift
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
import dftModel as DF
(fs, x) = UF.wavread('../../../sounds/ocean.wav')
(fs, x2) = UF.wavread('../../../sounds/impulse-response.wav')
x1 = x[40000:44096]
N = 4096

plt.figure(1, figsize=(9.5, 7))
plt.subplot(3,2,1)
plt.title('x1 (ocean.wav)')
plt.plot(x1, 'b')
plt.axis([0,N,min(x1),max(x1)])

plt.subplot(3,2,2)
plt.title('x2 (impulse-response.wav)')
plt.plot(x2, 'b')
plt.axis([0,N,min(x2),max(x2)])

mX1, pX1 = DF.dftAnal(x1, np.ones(N), N)
mX1 = mX1 - max(mX1)
plt.subplot(3,2,3)
plt.title('X1')
plt.plot(mX1, 'r')
plt.axis([0,N/2,-70,0])

mX2, pX2 = DF.dftAnal(x2, np.ones(N), N)
mX2 = mX2 - max(mX2)
plt.subplot(3,2,4)
plt.title('X2')
plt.plot(mX2, 'r')
plt.axis([0,N/2,-70,0])

y = np.convolve(x1, x2)
mY, pY = DF.dftAnal(y[0:N], np.ones(N), N)
mY = mY - max(mY)
plt.subplot(3,2,5)
plt.title('DFT(x1 * x2)')
plt.plot(mY, 'r')
plt.axis([0,N/2,-70,0])

plt.subplot(3,2,6)
plt.title('X1 x X2')
mY1 = 20*np.log10(np.abs(fft(x1) * fft(x2)))
mY1 = mY1 - max(mY1)
plt.plot(mY1[0:N//2], 'r')
plt.axis([0,N//2,-84,0])

plt.tight_layout()
plt.savefig('convolution-1.png')
plt.show()
