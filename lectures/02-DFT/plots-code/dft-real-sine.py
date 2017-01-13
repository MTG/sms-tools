import numpy as np
import matplotlib.pyplot as plt
tol = 1e-5

plt.figure(1, figsize=(9.5, 7))
N = 64
k0 = 7.5
X = np.array([])
nv = np.arange(-N/2, N/2)
kv = np.arange(-N/2, N/2)
x = np.cos(2*np.pi*k0/N*nv)

plt.subplot(311)
plt.title('x; k_0 = 7.5, N = 64')
plt.plot(nv, x,'b', lw=1.5)
plt.axis([-N/2,N/2-1,-1,1])
for k in kv:
	s = np.exp(1j*2*np.pi*k/N*nv)
	X = np.append(X, sum(x*np.conjugate(s)))

X.real[np.abs(X.real) < tol] = 0.0
X.imag[np.abs(X.imag) < tol] = 0.0

plt.subplot(312)
plt.title('magnitude spectrum: abs(X)')
plt.plot(kv, abs(X), 'r', lw=1.5)
plt.axis([-N/2,N/2-1,0,N])

plt.subplot(313)
plt.title('phase spectrum: angle(X)')
plt.plot(kv, np.angle(X),'c', lw=1.5)
plt.axis([-N/2,N/2-1,-np.pi,np.pi])

plt.tight_layout()
plt.savefig('dft-real-sine.png')
plt.show()
