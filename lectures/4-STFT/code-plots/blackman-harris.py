import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy import signal

M = 64
N = 512
hN = N/2     
hM = M/2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1)
fftbuffer[hN-hM:hN+hM]=signal.blackmanharris(M)
plt.subplot(2,1,1)
plt.plot(np.arange(-hN, hN), fftbuffer, 'b')
plt.axis([-hN, hN, 0, 1.1])
plt.xlabel('time (samples)')
plt.ylabel('amplitude')


X = fft(fftbuffer)
mX = 20*np.log10(abs(X)) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]      

plt.subplot(2,1,2)
plt.plot(np.arange(-hN, hN), mX1-max(mX), 'r')
plt.axis([-hN,hN,-110,0])
plt.xlabel('frequency (bins)')
plt.ylabel('amplitude (dB)')

plt.show()