import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
from scipy import signal

M = 32
N = 128
hN = N/2     
hM = M/2
fftbuffer = np.zeros(N)
w = signal.blackman(M)

plt.subplot(3,2,1)
plt.figure(1)

plt.plot(np.arange(-hM, hM), w, 'b')
plt.axis([-hM, hM-1, 0, 1])
plt.xlabel('time (samples)')
plt.ylabel('amplitude')
plt.title('M=32')

fftbuffer = np.zeros(N)                         
fftbuffer[:hM] = w[hM:] 
fftbuffer[N-hM:] = w[:hM]
X = fft(fftbuffer)
mX = 20*np.log10(abs(fftshift(X)))    
plt.subplot(3,2,3)
plt.plot(np.arange(-hN, hN), mX-max(mX), 'r')
plt.axis([-hN/2,hN/2,-80,0])
plt.xlabel('frequency (bins)')
plt.ylabel('amplitude (dB)')

pX = np.angle(fftshift(X))
plt.subplot(3,2,5)
plt.plot(np.arange(-hN, hN), pX, 'g')
plt.axis([-hN,hN-1,-np.pi,np.pi])
plt.xlabel('frequency (bins)')
plt.ylabel('phase (radians)')

M = 31
N = 128
hN = N/2     
hM = (M+1)/2
fftbuffer = np.zeros(N)
w = signal.blackman(M)

plt.subplot(3,2,2)
plt.plot(np.arange(-hM, hM-1), w, 'b')
plt.axis([-hM, hM, 0, 1])
plt.xlabel('time (samples)')
plt.ylabel('amplitude')
plt.title('M=31')

fftbuffer = np.zeros(N) 
fftbuffer[:hM] = w[hM-1:] 
fftbuffer[N-hM+1:] = w[:hM-1]                         
X = fft(fftbuffer)
mX = 20*np.log10(abs(fftshift(X)))    
plt.subplot(3,2,4)
plt.plot(np.arange(-hN, hN), mX-max(mX), 'r')
plt.axis([-hN/2,hN/2-1,-80,0])
plt.xlabel('frequency (bins)')
plt.ylabel('amplitude (dB)')

pX = np.angle(fftshift(X))
plt.subplot(3,2,6)
plt.plot(np.arange(-hN, hN), pX, 'g')
plt.axis([-hN,hN-1,-np.pi,np.pi])
plt.xlabel('frequency (bins)')
plt.ylabel('phase (radians)')
plt.show()