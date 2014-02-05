import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
import smsWavplayer as wp

N = 1024
hN = N/2
M = 601
hM = (M+1)/2
w = np.blackman(M)

(fs, x) = wp.wavread('oboe.wav')
xw = x[40000:40000+M] * w

plt.figure(1)
fftbuffer = np.zeros(N)                         
fftbuffer[:hM] = xw[hM-1:] 
fftbuffer[N-hM+1:] = xw[:hM-1]
plt.subplot(3,1,1)
plt.plot(np.arange(N), fftbuffer, 'b')
plt.axis([0, N, min(xw), max(xw)])
plt.title('fftbuffer')

X = fft(fftbuffer)
mX = 20*np.log10(abs(X[:hN]))       
plt.subplot(3,1,2)
plt.plot(fs*np.arange(hN)/float(N),mX, 'r')
plt.axis([250,3300,-20,max(mX)])
plt.title('Magintude spectrum: abs(X)')

pX = np.unwrap(np.angle(X[0:hN]))       
plt.subplot(3,1,3)
plt.plot(fs*np.arange(hN)/float(N),pX, 'g')
plt.axis([250,3300,-6,4])
plt.title('phase spectrum: angle(X)')

plt.show()