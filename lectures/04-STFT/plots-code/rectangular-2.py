import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

M = 64
N = 1024
hN = N//2     
hM = M//2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1, figsize=(7.5, 3.5))
fftbuffer[hN-hM:hN+hM]=np.ones(M)
plt.subplot(2,1,1)
plt.plot(np.arange(-hN, hN), fftbuffer, 'b', lw=1.5)
plt.axis([-hN, hN, 0, 1.1])


X = fft(fftbuffer)
mX = 20*np.log10(abs(X)) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]      

plt.subplot(2,1,2)
plt.plot(np.arange(-hN, hN), mX1-max(mX), 'r', lw=1.5)
plt.axis([-hN,hN,-40,0])

plt.tight_layout()
plt.savefig('rectangular-2.png')
plt.show()
