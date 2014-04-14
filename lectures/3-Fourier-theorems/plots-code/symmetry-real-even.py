import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from scipy.signal import triang
from scipy.fftpack import fft, fftshift


M = 127
N = 128
hM1 = int(math.floor((M+1)/2)) 
hM2 = int(math.floor(M/2)) 
x = triang(M)
fftbuffer = np.zeros(N)
fftbuffer[:hM1] = x[hM2:]
fftbuffer[N-hM2:] = x[:hM2] 
X = fftshift(fft(fftbuffer))
mX = abs(X)    
pX = np.unwrap(np.angle(X))

plt.figure(1, figsize=(9.5, 4))
plt.subplot(311)
plt.title('x[n]')
plt.plot(np.arange(-hM2, hM1, 1.0), x, 'b', lw=1.5)
plt.axis([-hM2, hM1, 0, 1])

plt.subplot(323)
plt.title('real(X)')
plt.plot(np.arange(-N/2, N/2, 1.0), np.real(X), 'r', lw=1.5)
plt.axis([-N/2, N/2, min(np.real(X)), max(np.real(X))])

plt.subplot(324)
plt.title('im(X)')
plt.plot(np.arange(-N/2, N/2, 1.0), np.imag(X), 'c', lw=1.5)
plt.axis([-N/2, N/2, -1, 1])     

plt.subplot(325)
plt.title('abs(X)')
plt.plot(np.arange(-N/2, N/2, 1.0), mX, 'r', lw=1.5)
plt.axis([-N/2,N/2,min(mX),max(mX)])  

plt.subplot(326)
plt.title('angle(X)')
plt.plot(np.arange(-N/2, N/2, 1.0), pX, 'c', lw=1.5)
plt.axis([-N/2, N/2, -1, 1])   

plt.tight_layout()
plt.savefig('symmetry-real-even.png')
plt.show()
