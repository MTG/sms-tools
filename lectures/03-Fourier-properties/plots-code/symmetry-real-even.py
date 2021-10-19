import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from scipy.signal import triang
from scipy.fftpack import fft, fftshift


M = 31
N = 31
hM1 = int(math.floor((M+1)/2)) 
hM2 = int(math.floor(M/2)) 
x = triang(M)
X = fftshift(fft(np.roll(x, 16)))
mX = abs(X)    
pX = np.unwrap(np.angle(X))

plt.figure(1, figsize=(9.5, 4))
plt.subplot(311)
plt.title('x[n]')
plt.plot(np.arange(-15, 16, 1.0), x, 'b-x', lw=1.5)
plt.axis([-15, 15, 0, 1.1])

plt.subplot(323)
plt.title('real(X)')
plt.plot(np.arange(-15, 16, 1.0), np.real(X), 'r-x', lw=1.5)
plt.axis([-15, 15, min(np.real(X)), max(np.real(X))])

plt.subplot(324)
plt.title('im(X)')
plt.plot(np.arange(-15, 16, 1.0), np.imag(X), 'c-x', lw=1.5)
plt.axis([-15, 15, -1, 1])     

plt.subplot(325)
plt.title('abs(X)')
plt.plot(np.arange(-15, 16, 1.0), mX, 'r-x', lw=1.5)
plt.axis([-15, 15,min(mX),max(mX)])  

plt.subplot(326)
plt.title('angle(X)')
plt.plot(np.arange(-15, 16, 1.0), pX, 'c-x', lw=1.5)
plt.axis([-15, 15, -1, 1])   

plt.tight_layout()
plt.savefig('symmetry-real-even.png')
plt.show()
