import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

M = 64
N = 1024
hN = N/2     
hM = M/2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1)
fftbuffer[hN-hM:hN+hM]=np.ones(M)
plt.subplot(3,1,1)
plt.plot(np.arange(-hN, hN), fftbuffer, 'b')
plt.axis([-hN, hN, 0, 1.1])
plt.xlabel('time (samples)')
plt.ylabel('amplitude')


X = fft(fftbuffer)
mX = 20*np.log10(abs(X)) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]      

plt.subplot(3,1,2)
plt.plot(np.arange(-hN, hN), mX1-max(mX), 'r')
plt.axis([-hN,hN,-40,0])
plt.xlabel('frequency (bins)')
plt.ylabel('amplitude (dB)')

n = np.arange(-hN, hN)
x = np.sin(2*np.pi*(n/float(N))*M/2.0)/np.sin(2*np.pi*(n/float(N))/2.0)
x1 = 20* np.log10(abs(x))
plt.subplot(3,1,3)
plt.plot(np.arange(-hN, hN), x1-max(x1), 'r')
plt.axis([-hN,hN,-40,0])
plt.xlabel('frequency (bins)')
plt.ylabel('amplitude (dB)')

plt.show()