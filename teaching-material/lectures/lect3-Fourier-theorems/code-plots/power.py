import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy.fftpack import fft, fftshift

plt.figure(1)
M= 64
N = 64
x = np.cos(2*np.pi*3/M*np.arange(M)) * np.hanning(M)

plt.subplot(2,1,1)
plt.title('x')
plt.plot(x, 'b')
plt.axis([0,M-1,-1,1])

powerx = sum(np.abs(x)**2)
print powerx

mX = np.abs(fftshift(fft(x, N)))
plt.subplot(2,1,2)
plt.title('abs(X)')
plt.plot(mX, 'r')
plt.axis([0,N-1,0,max(mX)+1])

powerX = sum(mX**2) / N

print powerX

plt.show()