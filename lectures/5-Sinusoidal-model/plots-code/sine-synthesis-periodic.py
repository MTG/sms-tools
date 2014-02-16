import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import ifft

Ns = 64
hNs = Ns/2

mY = np.zeros(hNs)
pY = np.zeros(hNs)
Y = np.zeros(Ns)
yw = np.zeros(Ns)
Y = np.zeros(Ns, dtype=complex)

k0 = 5
ph0 = 1.5
A0 = .8
mY[k0] = A0 * Ns
pY[k0] = ph0
Y[:hNs] = .5*mY*np.exp(1j*pY)
Y[hNs+1:] = Y[hNs-1:0:-1].conjugate()   

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(np.arange(hNs), mY, color='r')
plt.axis([0,hNs-1,-1,Ns])
plt.xlabel('frenquency (bins)')
plt.ylabel('amplitude')
plt.title('magnitude spectrum; k0 = 5, A0 = .8, N = 64')

plt.subplot(3,1,2)
plt.plot(np.arange(hNs), pY, color='c')
plt.axis([0,hNs-1,-.3, 2])
plt.xlabel('frenquency (bins)')
plt.ylabel('phase (radians)')
plt.title('phase spectrum; k0 = 5, theta0 = .1.5, N = 64')

y = np.real(ifft(Y))     
yw[:hNs-1] = y[hNs+1:] 
yw[hNs-1:] = y[:hNs+1] 

plt.subplot(3,1,3)
plt.plot(np.arange(Ns), yw, 'b')
plt.axis([0, Ns-1,-.9,.9])
plt.xlabel('time (samples)')
plt.ylabel('amplitude')
plt.title('synthesized sound; N = 64')

plt.show()
