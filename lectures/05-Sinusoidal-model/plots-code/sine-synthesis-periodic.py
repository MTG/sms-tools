import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import ifft

Ns = 64
hNs = Ns//2

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

plt.figure(1, figsize=(9, 5))
plt.subplot(3,1,1)
plt.plot(np.arange(hNs), mY, color='r', lw=1.5)
plt.axis([0,hNs-1,-1,Ns])
plt.title('mY; k0 = 5, A0 = .8, N = 64')

plt.subplot(3,1,2)
plt.plot(np.arange(hNs), pY, color='c', lw=1.5)
plt.axis([0,hNs-1,-.3, 2])
plt.title('pY; k0 = 5, theta0 = .1.5, N = 64')

y = np.real(ifft(Y))     
yw[:hNs-1] = y[hNs+1:] 
yw[hNs-1:] = y[:hNs+1] 

plt.subplot(3,1,3)
plt.plot(np.arange(Ns), yw, 'b', lw=1.5)
plt.axis([0, Ns-1,-.9,.9])
plt.title('y; N = 64')

plt.tight_layout()
plt.savefig('sine-synthesis-periodic.png')
plt.show()
