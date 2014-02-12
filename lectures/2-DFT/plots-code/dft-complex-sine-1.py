import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
N = 64
k0 = 7
X = np.array([])
x = np.exp(1j*2*np.pi*k0/N*np.arange(N))

plt.subplot(311)
plt.title('complex sinewave: x; k = 7, N = 64')
plt.plot(np.arange(N), np.real(x),'b')
plt.plot(np.arange(N), np.imag(x),'g')
plt.axis([0,N-1,-1,1])
for k in range(N):
	s = np.exp(1j*2*np.pi*k/N*np.arange(N))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(312)
plt.title('magnitude spectrum: abs(X)')
plt.plot(np.arange(N), abs(X), 'r')
plt.axis([0,N-1,0,N])

plt.subplot(313)
plt.title('phase spectrum: angle(X)')
plt.plot(np.arange(N), np.angle(X),'c')
plt.axis([0,N-1,-np.pi,np.pi])

plt.show()
