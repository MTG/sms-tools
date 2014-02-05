import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft
import math

(fs, x) = read('oboe.wav')
M = 400
start = .8*fs   
xp = x[start:start+M]/float(max(x[start:start+M]))                                          
z = np.correlate(xp,xp,'full')

plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(M)/float(fs), xp)
plt.axis([0, M/float(fs), min(xp), max(xp)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.title('oboe sound')

plt.subplot(212)
plt.plot(z[M-1:]/max(z),'r')
plt.axis([0, M, -.5, 1.0])
plt.xlabel('Hz')
plt.ylabel('correlation factor')
plt.title('Autocorrelation function')
plt.xticks([100,200,300,400], [fs/100.0, fs/200.0, fs/300.0, fs/400.0])
plt.show()