import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import stft as STFT
import sineModel as SM
import utilFunctions as UF

M = 256
N = 256
hN = N//2
hM = M//2
fs = 44100
f0 = 5000.0
A0 = 1
ph = 1.5
t = np.arange(-hM,hM)/float(fs)
x = A0 * np.cos(2*np.pi*f0*t+ph)

w = hamming(M)
xw = x*w
fftbuffer = np.zeros(N) 
fftbuffer[0:M] = xw
X = fft(fftbuffer) 
mX = abs(X)
pX = np.angle(X[0:hN])

powerX = sum(2*mX[0:hN]**2)/N

mask = np.zeros(N//2)
mask[int(N*f0/fs-2*N/float(M)):int(N*f0/fs+3*N/float(M))] = 1.0
mY = mask*mX[0:hN]
powerY = sum(2*mY[0:hN]**2)/N

Y = np.zeros(N, dtype = complex)
Y[:hN] = mY * np.exp(1j*pX) 
Y[hN+1:] = mY[:0:-1] * np.exp(-1j*pX[:0:-1]) 
 
y = ifft(Y)
SNR1 = -10*np.log10((powerX-powerY)/(powerX))

freqaxis = fs*np.arange(0,N/2)/float(N)
taxis = np.arange(N)/float(fs) 

plt.figure(1, figsize=(9, 6))
plt.subplot(3,2,1)
plt.plot(20*np.log10(mY[:hN])-max(20*np.log10(mY[:hN])), 'r', lw=1.5)
plt.title ('mX, mY (main lobe); Hamming')
plt.plot(20*np.log10(mX[:hN])-max(20*np.log10(mX[:hN])), 'r', lw=1.5, alpha=.2)
plt.axis([0,hN,-120,0])

plt.subplot(3,2,3)
plt.plot(y[0:M], 'b', lw=1.5)
plt.axis([0,M,-1,1])
plt.title ('y (synthesis of main lobe)')

plt.subplot(3,2,5)
yerror = xw - y
plt.plot(yerror, 'k', lw=1.5)
plt.axis([0,M,-.003,.003])
plt.title ("error function: x-y; SNR = ${%d}$ dB" %(SNR1))

w = blackmanharris(M)
xw = x*w
fftbuffer = np.zeros(N) 
fftbuffer[0:M] = xw
X = fft(fftbuffer) 
mX = abs(X) 
pX = np.angle(X[0:hN])

powerX = sum(2*mX[0:hN]**2)/N

mask = np.zeros(N//2)
mask[int(N*f0/fs-4*N/float(M)):int(N*f0/fs+5*N/float(M))] = 1.0
mY = mask*mX[0:hN]
powerY = sum(2*mY[0:hN]**2)/N

Y = np.zeros(N, dtype = complex)
Y[:hN] = mY * np.exp(1j*pX) 
Y[hN+1:] = mY[:0:-1] * np.exp(-1j*pX[:0:-1]) 
 
y = ifft(Y)
SNR2 = -10*np.log10((powerX-powerY)/(powerX))

plt.subplot(3,2,2)
plt.plot(20*np.log10(mY[:hN])-max(20*np.log10(mY[:hN])), 'r', lw=1.5)
plt.title ('mX, mY (main lobe); Blackman Harris')
plt.plot(20*np.log10(mX[:hN])-max(20*np.log10(mX[:hN])), 'r', lw=1.5, alpha=.2)
plt.axis([0,hN,-120,0])

plt.subplot(3,2,4)
plt.plot(y[0:M], 'b', lw=1.5)
plt.axis([0,M,-1,1])
plt.title ('y (synthesis of main lobe)')

plt.subplot(3,2,6)
yerror2 = xw - y
plt.plot(yerror2, 'k', lw=1.5)
plt.axis([0,M,-.003,.003])
plt.title ("error function: x-y; SNR = ${%d}$ dB" %(SNR2))

plt.tight_layout()
plt.savefig('spec-sine-synthesis-lobe.png')
plt.show()
