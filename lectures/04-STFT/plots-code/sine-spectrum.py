import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft

N = 256
M = 63
f0 = 1000
fs = 10000
A0 = .8 
hN = N//2 
hM = (M+1)//2
fftbuffer = np.zeros(N)
X1 = np.zeros(N, dtype='complex')
X2 = np.zeros(N, dtype='complex')

x = A0 * np.cos(2*np.pi*f0/fs*np.arange(-hM+1,hM))

plt.figure(1, figsize=(9.5, 7))
w = np.hanning(M)
plt.subplot(2,3,1)
plt.title('w (hanning window)')
plt.plot(np.arange(-hM+1, hM), w, 'b', lw=1.5)
plt.axis([-hM+1, hM, 0, 1])

fftbuffer[:hM] = w[hM-1:]
fftbuffer[N-hM+1:] = w[:hM-1]  
X = fft(fftbuffer)
X1[:hN] = X[hN:]
X1[N-hN:] = X[:hN]
mX = 20*np.log10(abs(X1))       

plt.subplot(2,3,2)
plt.title('mW')
plt.plot(np.arange(-hN, hN), mX, 'r', lw=1.5)
plt.axis([-hN,hN,-40,max(mX)])

pX = np.angle(X1)
plt.subplot(2,3,3)
plt.title('pW')
plt.plot(np.arange(-hN, hN), np.unwrap(pX), 'c', lw=1.5)
plt.axis([-hN,hN,min(np.unwrap(pX)),max(np.unwrap(pX))])

plt.subplot(2,3,4)
plt.title('xw (windowed sinewave)')
xw = x*w
plt.plot(np.arange(-hM+1, hM), xw, 'b', lw=1.5)
plt.axis([-hM+1, hM, -1, 1])

fftbuffer = np.zeros(N)
fftbuffer[0:hM] = xw[hM-1:]
fftbuffer[N-hM+1:] = xw[:hM-1]
X = fft(fftbuffer)
X2[:hN] = X[hN:]
X2[N-hN:] = X[:hN]
mX2 = 20*np.log10(abs(X2))  

plt.subplot(2,3,5)
plt.title('mXW')
plt.plot(np.arange(-hN, hN), mX2, 'r', lw=1.5)
plt.axis([-hN,hN,-40,max(mX)])

pX = np.angle(X2)
plt.subplot(2,3,6)
plt.title('pXW')
plt.plot(np.arange(-hN, hN), np.unwrap(pX), 'c', lw=1.5)
plt.axis([-hN,hN,min(np.unwrap(pX)),max(np.unwrap(pX))])

plt.tight_layout()
plt.savefig('sine-spectrum.png')
plt.show()
