import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft

(fs, x) = read('oboe.wav')
N = 256
start = .8*fs
L = 3
H = 128
plt.figure(1)
plt.plot(np.arange(0, L*(H+1), 1.0)/fs, x[start:start+L*(H+1)], 'b')
plt.axis([0, L*(H+1.0)/fs, min(x[start:start+L*(H+1)]), max(x[start:start+L*(H+1)])])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')

plt.figure(2)
xw = x[start:start+N] * np.hamming(N)
plt.subplot(1,2,1)
plt.plot(np.arange(0, N, 1.0)/fs, xw, 'b')
plt.axis([0, N/float(fs), min(x[start:start+L*(H+1)]), max(x[start:start+L*(H+1)])])

X = fft(xw)
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X))
plt.subplot(1,2,2)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2], 'r')
plt.axis([0,fs/2.0,0,max(mX)])

plt.figure(3)
start = start + H
xw = x[start:start+N] * np.hamming(N)
plt.subplot(1,2,1)
plt.plot(np.arange(0, N, 1.0)/fs, xw, 'b')
plt.axis([0, N/float(fs), min(x[start:start+L*(H+1)]), max(x[start:start+L*(H+1)])])

X = fft(xw)
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X))
plt.subplot(1,2,2)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2], 'r')
plt.axis([0,fs/2.0,0,max(mX)])

plt.figure(4)
start = start + H
xw = x[start:start+N] * np.hamming(N)
plt.subplot(1,2,1)
plt.plot(np.arange(0, N, 1.0)/fs, xw, 'b')
plt.axis([0, N/float(fs), min(x[start:start+L*(H+1)]), max(x[start:start+L*(H+1)])])

X = fft(xw)
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X))
plt.subplot(1,2,2)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2], 'r')
plt.axis([0,fs/2.0,0,max(mX)])

plt.show()