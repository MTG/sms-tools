import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy import signal

M = 64
N = 512
hN = N//2     
hM = M//2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1, figsize=(7.5, 4))
fftbuffer[hN-hM:hN+hM]=signal.blackman(M)
plt.subplot(2,1,1)
plt.plot(np.arange(-hN, hN), fftbuffer, 'b', lw=1.5)
plt.axis([-hN, hN, 0, 1.1])


X = fft(fftbuffer)
mX = 20*np.log10(abs(X)) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]      

plt.subplot(2,1,2)
plt.plot(np.arange(-hN, hN), mX1-max(mX), 'r', lw=1.5)
plt.axis([-hN,hN,-80,0])

plt.tight_layout()
plt.savefig('blackman.png')
plt.show()
