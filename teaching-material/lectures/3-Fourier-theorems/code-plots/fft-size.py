import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft

(fs, x) = read('oboe.wav')
M = 512
N = 512
start = .8*fs
xw = x[start:start+M] * np.hamming(M)
plt.figure(1)
plt.subplot(311)
plt.plot(np.arange(start, (start+M), 1.0)/fs, xw)
plt.axis([start/fs, (start+M)/fs, min(xw), max(xw)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
X = fft(xw, N)
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X)) 
plt.subplot(312)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2])
plt.axis([0,fs/2.0,0,max(mX)])
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude (dB)')

M = 512
N = 1024
start = .8*fs
xw = x[start:start+M] * np.hamming(M)
X = fft(xw, N)
mX = 20 * np.log10(abs(X)/N)        
plt.subplot(313)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2])
plt.axis([0,fs/2.0,0,max(mX)])
plt.xlabel('frequency (Hz)')
plt.show()