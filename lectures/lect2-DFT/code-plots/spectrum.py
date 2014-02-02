import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft

(fs, x) = read('oboe.wav')
N = 256
start = .8*fs
xw = x[start:start+N] * np.hamming(N)
plt.figure(1)
plt.subplot(311)
plt.plot(np.arange(start, (start+N), 1.0)/fs, xw)
plt.axis([start/fs, (start+N)/fs, min(xw), max(xw)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
X = fft(xw)
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X)) 
plt.subplot(312)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX[0:N/2], 'r')
plt.axis([0,fs/2.0,0,max(mX)])
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude (dB)')
plt.subplot(313)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), pX[:N/2], 'c')
plt.axis([0,fs/2.0,-28,max(pX)])
plt.xlabel('frequency (Hz)')
plt.ylabel('phase (radians)')
plt.show()