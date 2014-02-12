import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read
from scipy.fftpack import fft, fftshift

(fs, x) = read('../../../sounds/oboe-A4.wav')
N = 512
M = 401
hN = N/2 
hM = (M+1)/2     
start = .8*fs
xw = x[start-hM:start+hM-1] * np.hamming(M)
plt.figure(1)
plt.subplot(411)
plt.plot(np.arange(-hM, hM-1), xw)
plt.axis([-hN, hN-1, min(xw), max(xw)])
plt.ylabel('amplitude')
plt.title('input signal: x = read(oboe-A4.wav), M = 401')

fftbuffer = np.zeros(N)                         
fftbuffer[:hM] = xw[hM-1:] 
fftbuffer[N-hM+1:] = xw[:hM-1]        
plt.subplot(412)
plt.plot(np.arange(0, N), fftbuffer)
plt.axis([0, N, min(xw), max(xw)])
plt.ylabel('amplitude')
plt.title('fftbuffer: N = 512')

X = fftshift(fft(fftbuffer))
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X)) 
plt.subplot(413)
plt.plot(np.arange(-hN, hN), mX, 'r')
plt.axis([-hN,hN-1,-5,max(mX)])
plt.ylabel('amplitude (dB)')
plt.title('mag spectrum')

plt.subplot(414)
plt.plot(np.arange(-hN, hN), pX, 'c')
plt.axis([-hN,hN-1,min(pX),max(pX)])
plt.ylabel('phase (radians)')
plt.title('phase spectrum')
plt.show()
