import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
N = 64
k0 = 7.5
X = np.array([])
y = np.array([])
plt.figure(1)
x = np.exp(1j*2*np.pi*k0/N*np.arange(N))
for k in range(N):
	s = np.exp((1j*2*np.pi*k/N)*np.arange(N))
	X = np.append(X, sum(x*np.conjugate(s)))
plt.subplot(411)
plt.title('complex spectrum: X')
plt.plot(np.arange(N), np.real(X))
plt.plot(np.arange(N), np.imag(X))
plt.axis([0,N-1,-N,N])
plt.subplot(412)
plt.title('magnitude spectrum: abs(X)')
plt.plot(np.arange(N), abs(X))
plt.axis([0,N-1,0,N+5])
plt.subplot(413)
plt.title('phase spectrum: angle(X)')
plt.plot(np.arange(N), np.angle(X))
plt.axis([0,N-1,-np.pi,np.pi])
for n in range(N):
	s = np.exp((1j*2*np.pi*n/N)*np.arange(N))
	y = np.append(y, sum(X*s)/N)
plt.subplot(414)
plt.title('inverse spectrum: IDFT(X)')
plt.plot(np.arange(N), np.real(y))
plt.plot(np.arange(N), np.imag(y))
plt.axis([0,N-1,-1,1])
plt.show()
