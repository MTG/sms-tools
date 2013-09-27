import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft

(fs, x) = read('oboe.wav')
M = 256
H = 128
start = .8*fs

plt.figure(1)
x = x[start:start+4*M]
plt.plot(np.arange(start, (start+4*M), 1.0)/fs, x)
# plt.axis([start/fs, (start+4*M)/fs, -1, max(x)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')

start +=M
offset = 1
x1 = ones(4*M)
x1[start:start+M] += x[start:start+M] * np.hamming(M)
plt.plot(np.arange(start, (start+4*M), 1.0)/fs, x1)
# plt.axis([start/fs, (start+4*M)/fs, min(x), max(x)])

plt.show()