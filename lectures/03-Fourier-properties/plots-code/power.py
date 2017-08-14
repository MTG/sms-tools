import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift

plt.figure(1, figsize=(6, 5))
M= 64
N = 64
x = np.cos(2*np.pi*3/M*np.arange(M)) * np.hanning(M)

plt.subplot(2,1,1)
plt.title('x')
plt.plot(np.arange(-M/2.0,M/2), x, 'b', lw=1.5)
plt.axis([-M/2,M/2-1,-1,1])

powerx = sum(np.abs(x)**2)
print (powerx)

mX = np.abs(fftshift(fft(x, N)))
plt.subplot(2,1,2)
plt.title('abs(X)')
plt.plot(np.arange(-N/2.0,N/2), mX, 'r', lw=1.5)
plt.axis([-N/2,N/2-1,0,max(mX)])

powerX = sum(mX**2) / N
print (powerX)

plt.tight_layout()
plt.savefig('power.png')
plt.show()
