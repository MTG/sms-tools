import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft
import math

(fs, x) = read('oboe.wav')
M = 651
N = 1024
hN = N/2  
start = .8*fs                                             
w = np.blackman(M)
hM1 = int(math.floor((w.size+1)/2))                     
hM2 = int(math.floor(w.size/2))                         
fftbuffer = np.zeros(N)                                 
xw = x[start:start+M] * w   
fftbuffer[:hM1] = xw[hM2:]
fftbuffer[N-hM2:] = xw[:hM2]        
X = fft(fftbuffer)
mX = 20 * np.log10( abs(X[:hN]) )  
pX = np.unwrap( np.angle(X[:hN]) ) 

plt.figure(1)
plt.subplot(311)
xp = x[start:start+M]/float(max(x[start:start+M]))
plt.plot(np.arange(start, (start+M), 1.0)/fs, xp)
plt.axis([start/fs, (start+M)/fs, min(xp), max(xp)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.title('oboe sound')

plt.subplot(312)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX-max(mX))
plt.axis([0,fs/8.0,-55,0])
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude (dB)')
plt.subplot(313)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), pX)
plt.axis([0,fs/8.0,-5,15])
plt.xlabel('frequency (Hz)')
plt.ylabel('phase (radians)')
plt.show()