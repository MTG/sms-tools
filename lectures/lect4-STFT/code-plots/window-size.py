import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft

(fs, x) = read('oboe.wav')
N = 128
start = .8*fs
xw = x[start:start+N] * np.hamming(N)
plt.figure(1)
plt.subplot(321)
plt.plot(np.arange(start, (start+N), 1.0)/fs, xw)
plt.axis([start/fs, (start+N)/fs, min(xw), max(xw)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
X = fft(xw)
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X)) 
plt.subplot(323)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2])
plt.axis([0,fs/2.0,0,max(mX)])
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude (dB)')
plt.subplot(325)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), pX[:N/2])
plt.axis([0,fs/2.0,-28,max(pX)])
plt.xlabel('frequency (Hz)')
plt.ylabel('phase (radians)')

N = 1024
start = .8*fs
xw = x[start:start+N] * np.hamming(N)
plt.subplot(322)
plt.plot(np.arange(start, (start+N), 1.0)/fs, xw)
plt.axis([start/fs, (start+N)/fs, min(xw), max(xw)])
plt.xlabel('time (sec)')
X = fft(xw)
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X)) 
plt.subplot(324)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2])
plt.axis([0,fs/2.0,0,max(mX)])
plt.xlabel('frequency (Hz)')
plt.subplot(326)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), pX[:N/2])
plt.axis([0,fs/2.0,-28,max(pX)])
plt.xlabel('frequency (Hz)')
plt.show()