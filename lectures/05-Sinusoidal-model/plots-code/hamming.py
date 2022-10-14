import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
eps = np.finfo(float).eps

M = 64
N = 512
hN = N//2     
hM = M//2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1, figsize=(4, 6))
fftbuffer[hN-hM:hN+hM]=np.hamming(M)
plt.subplot(2,1,1)
plt.plot(np.arange(-hN+1, hN+1), fftbuffer, 'b', lw=1.5)
plt.axis([-hM+1, hM, 0, 1])
plt.title('w (hamming window), M=64')


X = fft(fftbuffer)
absX = abs(X)
absX[absX < eps] = eps
mX = 20*np.log10(absX) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]      

plt.subplot(2,1,2)
plt.plot(M*np.arange(-hN, hN)/float(N), mX1-max(mX), 'r', lw=1.5)
plt.axis([-hM,hM-1,-55,1])
plt.title('mW')

plt.tight_layout()
plt.savefig('hamming.png')
plt.show()
