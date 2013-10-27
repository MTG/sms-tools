import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft
import math

(fs, x) = read('tibetan-chant.wav')
M = 4501
N = 8192
hN = N/2  
start = 5.0*fs                                             
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
plt.title('tibetan sound')

plt.subplot(312)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), mX-max(mX))
plt.axis([0,fs/50.0,-70,0])
plt.xlabel('frequency (Hz)')
plt.ylabel('amplitude (dB)')
plt.subplot(313)
plt.plot(np.arange(0, fs/2.0, float(fs)/N), pX)
plt.axis([0,fs/50.0,3,18])
plt.xlabel('frequency (Hz)')
plt.ylabel('phase (radians)')
plt.show()