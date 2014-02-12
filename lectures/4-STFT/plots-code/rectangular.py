import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft

M = 64
N = 1024
hN = N/2     
hM = M/2
fftbuffer = np.zeros(N)
mX1 = np.zeros(N)

plt.figure(1)
fftbuffer[hN-hM:hN+hM]=np.ones(M)
plt.subplot(2,1,1)
plt.plot(np.arange(-hN, hN), fftbuffer, 'b')
plt.axis([-hN, hN, 0, 1.1])
plt.title('Rectangular window, M = 64')


X = fft(fftbuffer)
mX = 20*np.log10(abs(X)) 
mX1[:hN] = mX[hN:]
mX1[N-hN:] = mX[:hN]      

plt.subplot(2,1,2)
plt.plot(np.arange(-hN, hN), mX1-max(mX), 'r')
plt.axis([-hN,hN,-40,0])
plt.title('magnitude spectrum, N = 1024')
plt.annotate('main-lobe', xy=(0,-10), xytext=(-200, -5), fontsize=16, arrowprops=(dict(facecolor='black', width=2, headwidth=6, shrink=0.01)))
plt.annotate('highest side-lobe', xy=(32,-13), xytext=(100, -10), fontsize=16, arrowprops=(dict(facecolor='black', width=2, headwidth=6, shrink=0.01)))


#n = np.arange(-hN, hN)
#x = np.sin(2*np.pi*(n/float(N))*M/2.0)/np.sin(2*np.pi*(n/float(N))/2.0)
#x1 = 20* np.log10(abs(x))
#plt.subplot(3,1,3)
#plt.plot(np.arange(-hN, hN), x1-max(x1), 'r')
#plt.axis([-hN,hN,-40,0])
#plt.xlabel('frequency (bins)')
#plt.ylabel('amplitude (dB)')

plt.show()
