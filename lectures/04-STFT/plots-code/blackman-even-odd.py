import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
from scipy import signal

M = 32
N = 128
hN = N//2     
hM = M//2
fftbuffer = np.zeros(N)
w = signal.blackman(M)

plt.figure(1, figsize=(9.5, 6))

plt.subplot(3,2,1)
plt.plot(np.arange(-hM, hM), w, 'b', lw=1.5)
plt.axis([-hM, hM-1, 0, 1.05])
plt.title('w1, M=32')

fftbuffer = np.zeros(N)                         
fftbuffer[:hM] = w[hM:] 
fftbuffer[N-hM:] = w[:hM]
X = fft(fftbuffer)
mX = 20*np.log10(abs(fftshift(X)))    
plt.subplot(3,2,3)
plt.plot(np.arange(-hN, hN), mX-max(mX), 'r', lw=1.5)
plt.axis([-hN//2,hN//2,-80,1])
plt.title('mW1')

pX = np.angle(fftshift(X))
plt.subplot(3,2,5)
plt.plot(np.arange(-hN, hN), pX, 'c', lw=1.5)
plt.axis([-hN,hN-1,-np.pi,np.pi])
plt.title('pW1')

M = 31
N = 128
hN = N//2     
hM = (M+1)//2
fftbuffer = np.zeros(N)
w = signal.blackman(M)

plt.subplot(3,2,2)
plt.plot(np.arange(-hM, hM-1), w, 'b', lw=1.5)
plt.axis([-hM, hM, 0, 1.05])
plt.title('w2, M=31')

fftbuffer = np.zeros(N) 
fftbuffer[:hM] = w[hM-1:] 
fftbuffer[N-hM+1:] = w[:hM-1]                         
X = fft(fftbuffer)
mX = 20*np.log10(abs(fftshift(X)))    
plt.subplot(3,2,4)
plt.plot(np.arange(-hN, hN), mX-max(mX), 'r', lw=1.5)
plt.axis([-hN/2,hN/2-1,-80,1])
plt.title('mW2')

pX = np.angle(fftshift(X))
plt.subplot(3,2,6)
plt.plot(np.arange(-hN, hN), pX, 'c', lw=1.5)
plt.axis([-hN,hN-1,-np.pi,np.pi])
plt.title('pW2')

plt.tight_layout()
plt.savefig('blackman-even-odd.png')
plt.show()
