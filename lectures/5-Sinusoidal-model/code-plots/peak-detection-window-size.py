import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy import signal

N = 1024
hN = N/2
f0 = 400.0
fs = 10000.0
M = 8 * fs / f0
w = np.blackman(M)

x = signal.sawtooth(2 * np.pi * (f0/fs) * np.arange(M))

plt.figure(1)
plt.subplot(4,1,1)
plt.plot(np.arange(M)/fs, x, 'b')
plt.axis([0, M/fs, -1, 1])
plt.title('x: f0 = 400, fs = 10000')

plt.subplot(4,1,2)
plt.plot(np.arange(M), w, 'b')
plt.axis([0, M-1, 0, 1])
plt.title('blackman window: M = 8*fs/400 = 200')

xw = x * w
plt.subplot(4,1,3)
plt.plot(np.arange(M)/fs, xw, 'b')
plt.axis([0, M/fs, -1, 1])
plt.title('xw = x*w')
  
X = fft(xw, N)
mX = 20*np.log10(abs(X[0:hN]))       
plt.subplot(4,1,4)
plt.plot(fs*np.arange(hN)/N,mX-max(mX), 'r')
plt.axis([0,fs/2,-90,0])
plt.title('Magintude spectrum: abs(X)')

plt.show()