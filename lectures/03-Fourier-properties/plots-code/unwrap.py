import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift
import sys

sys.path.append('../../../software/models/')
import utilFunctions as UF

(fs, x) = UF.wavread('../../../sounds/soprano-E4.wav')
N = 1024
x1 = np.blackman(N)*x[40000:40000+N]

plt.figure(1, figsize=(9.5, 6))
X = np.array([])
x2 = np.zeros(N)

plt.subplot(4,1,1)
plt.title ('x (soprano-E4.wav)')
plt.plot(x1, lw=1.5)
plt.axis([0,N,min(x1),max(x1)])

X = fft(fftshift(x1))
mX = 20*np.log10(np.abs(X[0:N//2]))
pX = np.angle(X[0:N//2])

plt.subplot(4,1,2)
plt.title ('mX = magnitude spectrum')
plt.plot(mX, 'r', lw=1.5)
plt.axis([0,N/2,min(mX), max(mX)])

plt.subplot(4,1,3)
plt.title ('pX1 = phase spectrum')
plt.plot(pX, 'c', lw=1.5)
plt.axis([0,N/2,min(pX),max(pX)])

pX1 = np.unwrap(pX)
plt.subplot(4,1,4)
plt.title ('pX2: unwrapped phase spectrum')
plt.plot(pX1, 'c', lw=1.5)
plt.axis([0,N/2,min(pX1),max(pX1)])

plt.tight_layout()
plt.savefig('unwrap.png')
plt.show()
