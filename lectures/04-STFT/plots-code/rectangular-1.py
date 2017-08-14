import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

M = 64
N = 1024
hN = N//2     
hM = M//2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1, figsize=(9.5, 6))
fftbuffer[hN-hM:hN+hM]=np.ones(M)
plt.subplot(2,1,1)
plt.plot(np.arange(-hN, hN), fftbuffer, 'b', lw=1.5)
plt.axis([-hN, hN, 0, 1.1])
plt.title('w (rectangular window), M = 64')


X = fft(fftbuffer)
mX = 20*np.log10(abs(X)) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]      

plt.subplot(2,1,2)
plt.plot(np.arange(-hN, hN), mX1-max(mX), 'r', lw=1.5)
plt.axis([-hN,hN,-40,0])
plt.title('mW, N = 1024')
plt.annotate('main-lobe', xy=(0,-10), xytext=(-200, -5), fontsize=16, arrowprops=(dict(facecolor='black', width=2, headwidth=6, shrink=0.01)))
plt.annotate('highest side-lobe', xy=(32,-13), xytext=(100, -10), fontsize=16, arrowprops=(dict(facecolor='black', width=2, headwidth=6, shrink=0.01)))


plt.tight_layout()
plt.savefig('rectangular-1.png')
plt.show()
