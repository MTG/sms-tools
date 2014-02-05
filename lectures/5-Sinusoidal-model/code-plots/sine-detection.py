import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft

N = 1024
M1 = 221
M2 = 601
f0 = 1000
f1 = 1100
fs = 10000
A0 = .8
A1 = .5
hN = N/2                                                # size of positive spectrum
w1 = np.hanning(M1)
w2 = np.hanning(M2)

x = A0*np.cos(2*np.pi*f0/fs*np.arange(N)) + A1*np.cos(2*np.pi*f1/fs*np.arange(N))

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(np.arange(N), x, 'b')
plt.axis([0, N, -1.5, 1.5])
plt.title('sum of two sines: f0=1000, f1=1100')
  
X = fft(x[0:M1]*w1, N)
mX = 20*np.log10(abs(X[0:hN]))       
plt.subplot(3,1,2)
plt.plot((np.arange(hN)/float(N))*fs, mX-max(mX), 'r')
plt.axis([0,fs/2,-65,0])
plt.title('Magintude spectrum: M=221, fs=10000, N=1024')

X = fft(x[0:M2]*w2, N)
mX = 20*np.log10(abs(X[0:hN]))       
plt.subplot(3,1,3)
plt.plot((np.arange(hN)/float(N))*fs, mX-max(mX), 'r')
plt.axis([0,fs/2,-65,0])
plt.title('Magnitude spectrum: M=601, fs=10000, N=1024')

plt.show()