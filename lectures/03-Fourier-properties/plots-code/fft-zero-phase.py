import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
import sys

sys.path.append('../../../software/models/')
import utilFunctions as UF

(fs, x) = UF.wavread('../../../sounds/oboe-A4.wav')
N = 512
M = 401
hN = N//2 
hM = (M+1)//2     
start = int(.8*fs)
xw = x[start-hM:start+hM-1] * np.hamming(M)

plt.figure(1, figsize=(9.5, 6.5))
plt.subplot(411)
plt.plot(np.arange(-hM, hM-1), xw, lw=1.5)
plt.axis([-hN, hN-1, min(xw), max(xw)])
plt.title('x (oboe-A4.wav), M = 401')

fftbuffer = np.zeros(N)                         
fftbuffer[:hM] = xw[hM-1:] 
fftbuffer[N-hM+1:] = xw[:hM-1]        
plt.subplot(412)
plt.plot(np.arange(0, N), fftbuffer, lw=1.5)
plt.axis([0, N, min(xw), max(xw)])
plt.title('fftbuffer: N = 512')

X = fftshift(fft(fftbuffer))
mX = 20 * np.log10(abs(X)/N)        
pX = np.unwrap(np.angle(X)) 
plt.subplot(413)
plt.plot(np.arange(-hN, hN), mX, 'r', lw=1.5)
plt.axis([-hN,hN-1,-100,max(mX)])
plt.title('mX')

plt.subplot(414)
plt.plot(np.arange(-hN, hN), pX, 'c', lw=1.5)
plt.axis([-hN,hN-1,min(pX),max(pX)])
plt.title('pX')

plt.tight_layout()
plt.savefig('fft-zero-phase.png')
plt.show()
