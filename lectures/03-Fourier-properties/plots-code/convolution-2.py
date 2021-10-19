import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift

plt.figure(1, figsize=(9.5, 7))
M = 32
N = 32
x1 = 2*np.hanning(M)
x2 = np.cos(2*np.pi*3/M*np.arange(M))
y1 = (x1/sum(x1))*x2
mY1 = 20 * np.log10(np.abs(fftshift(fft(y1, N))))

plt.subplot(3,2,1)
plt.title('x1 (hanning)')
plt.plot(np.arange(-M/2, M/2), x1/2, 'b-x', lw=1.5)
plt.axis([-M/2,M/2,0,1])

plt.subplot(3,2,2)
plt.title('x2 (cosine)')
plt.plot(np.arange(-M/2, M/2),x2, 'b-x', lw=1.5)
plt.axis([-M/2,M/2,-1,1])

mX1 = 20 * np.log10(np.abs(fftshift(fft(x1, M)))/M)
plt.subplot(3,2,3)
plt.title('mX1')
plt.plot(np.arange(-N/2, N/2),mX1, 'r-x', lw=1.5)
plt.axis([-N/2,N/2,-80,0])

mX2 = 20 * np.log10(np.abs(fftshift(fft(x2, M)))/M)
mX2 = mX2 - max(mX2)
plt.subplot(3,2,4)
plt.title('mX2')
plt.plot(np.arange(-N/2, N/2),mX2, 'r-x', lw=1.5)
plt.axis([-N/2,N/2,-80,0])

plt.subplot(3,2,5)
plt.title('magnitude of DFT(x1 x x2)')
plt.plot(np.arange(-N/2, N/2),mY1+4, 'r-x', lw=1.5)
plt.axis([-N/2,N/2,-80,0])

Y2 = np.convolve(fftshift(fft((x1 / sum(x1)), M)), fftshift(fft(x2, M)))
mY2 = 20 * np.log10(np.abs(Y2))
mY2 = mY2 - max(mY2)

plt.subplot(3,2,6)
plt.title('magnitude of (X1 * X2)')
plt.plot(np.arange(-N//2, N//2),mY2[M//2:M+M//2], 'r-x', lw=1.5)
plt.axis([-N/2,N/2,-80,0])

plt.tight_layout()
plt.savefig('convolution-2.png')
plt.show()
