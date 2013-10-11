import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift

plt.figure(1)
M = 8
N1 = 8
N2 = 16
N3 = 32
x = np.cos(2*np.pi*2/M*np.arange(M)) * np.hanning(M)

plt.subplot(4,1,1)
plt.title('x, M=8')
plt.plot(x, 'b')
plt.axis([0,M-1,-1,1])

mX = 20 * np.log10(np.abs(fftshift(fft(x, N1))))
plt.subplot(4,1,2)
plt.plot(mX, 'ro')
plt.axis([0,N1-1,-20,max(mX)+1])
plt.title('abs(X), N=8')

mX = 20 * np.log10(np.abs(fftshift(fft(x, N2))))
plt.subplot(4,1,3)
plt.plot(mX, 'ro')
plt.axis([0,N2-1,-20,max(mX)+1])
plt.title('abs(X), N=16')

mX = 20 * np.log10(np.abs(fftshift(fft(x, N3))))
plt.subplot(4,1,4)
plt.plot(mX, 'ro')
plt.axis([0,N3-1,-20,max(mX)+1])
plt.title('abs(X), N=32')

plt.show()