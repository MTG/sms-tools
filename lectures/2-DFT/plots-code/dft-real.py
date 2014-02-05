import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
N = 64
k0 = 7.5
X = np.array([])
nv = np.arange(-N/2, N/2)
kv = np.arange(-N/2, N/2)
x = np.cos(2*np.pi*k0/N*nv)

plt.subplot(411)
plt.title('real sinewave: x')
plt.plot(nv, np.real(x),'b')
plt.axis([-N/2,N/2-1,-1,1])
for k in kv:
	s = np.exp(1j*2*np.pi*k/N*nv)
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(412)
plt.title('complex spectrum: X')
plt.plot(kv, np.real(X),'r')
plt.plot(kv, np.imag(X),'c')
plt.axis([-N/2,N/2-1,-N,N])

plt.subplot(413)
plt.title('magnitude spectrum: abs(X)')
plt.plot(kv, abs(X), 'r')
plt.axis([-N/2,N/2-1,0,N])

plt.subplot(414)
plt.title('phase spectrum: angle(X)')
plt.plot(kv, np.angle(X),'c')
plt.axis([-N/2,N/2-1,-np.pi,np.pi])

plt.show()
