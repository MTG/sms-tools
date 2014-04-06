import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift

plt.figure(1, figsize=(9.5, 6))
M = 8
N1 = 8
N2 = 16
N3 = 32
x = np.cos(2*np.pi*2/M*np.arange(M)) * np.hanning(M)

plt.subplot(4,1,1)
plt.title('x, M=8')
plt.plot(np.arange(-M/2.0,M/2), x, 'b', marker='x', lw=1.5)
plt.axis([-M/2,M/2-1,-1,1])

mX = 20 * np.log10(np.abs(fftshift(fft(x, N1))))
plt.subplot(4,1,2)
plt.plot(np.arange(-N1/2.0,N1/2), mX, marker='x', color='r', lw=1.5)
plt.axis([-N1/2,N1/2-1,-20,max(mX)+1])
plt.title('magnitude spectrum: mX1, N=8')

mX = 20 * np.log10(np.abs(fftshift(fft(x, N2))))
plt.subplot(4,1,3)
plt.plot(np.arange(-N2/2.0,N2/2),mX,marker='x',color='r', lw=1.5)
plt.axis([-N2/2,N2/2-1,-20,max(mX)+1])
plt.title('magnitude spectrum: mX2, N=16')

mX = 20 * np.log10(np.abs(fftshift(fft(x, N3))))
plt.subplot(4,1,4)
plt.plot(np.arange(-N3/2.0,N3/2),mX,marker='x',color='r', lw=1.5)
plt.axis([-N3/2,N3/2-1,-20,max(mX)+1])
plt.title('magnitude spectrum: mX3, N=32')

plt.tight_layout()
plt.savefig('zero-padding.png')
plt.show()
