import numpy as np
import matplotlib.pyplot as plt

N = 64
k0 = 10.5

plt.figure(1)
X = np.array([])
x = np.cos(2*np.pi*k0/N*np.arange(-N/2, N/2+1))

plt.subplot(4,2,1)
plt.title ('x1')
plt.plot(np.arange(-N/2, N/2+1), x)
plt.axis([-N/2,N/2,-1,1])

for k in range(-N/2, N/2+1):
	s = np.exp(1j*2*np.pi*k/N*np.arange(-N/2, N/2+1))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(4,2,3)
plt.title ('complex spectrum: X1')
plt.plot(np.arange(-N/2, N/2+1), np.real(X))
plt.plot(np.arange(-N/2, N/2+1), np.imag(X))
plt.axis([-N/2,N/2,-25,25])

plt.subplot(4,2,5)
plt.title ('magnitude spectrum: abs(X1)')
plt.plot(np.arange(-N/2, N/2+1), abs(X))
plt.axis([-N/2,N/2,0,25])

plt.subplot(4,2,7)
plt.title ('phase spectrum: angle(X1)')
plt.plot(np.arange(-N/2, N/2+1), np.unwrap(np.angle(X)))
plt.axis([-N/2, N/2,min(np.unwrap(np.angle(X))),max(np.unwrap(np.angle(X)))])

X = np.array([])
x = np.cos(2*np.pi*k0/N*np.arange(-N/2, N/2+1)+np.pi/2)

plt.subplot(4,2,2)
plt.title ('x2 (x1 shifted by +pi/2)')
plt.plot(np.arange(-N/2, N/2+1), x)
plt.axis([-N/2,N/2,-1,1])

for k in range(-N/2, N/2+1):
	s = np.exp(1j*2*np.pi*k/N*np.arange(-N/2, N/2+1))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(4,2,4)
plt.title ('complex spectrum: X2')
plt.plot(np.arange(-N/2, N/2+1), np.real(X))
plt.plot(np.arange(-N/2, N/2+1), np.imag(X))
plt.axis([-N/2,N/2,-25,25])

plt.subplot(4,2,6)
plt.title ('magnitude spectrum: abs(X2)')
plt.plot(np.arange(-N/2, N/2+1), abs(X))
plt.axis([-N/2,N/2,0,25])

plt.subplot(4,2,8)
plt.title ('phase spectrum: angle(X2)')
plt.plot(np.arange(-N/2, N/2+1), np.unwrap(np.angle(X)))
plt.axis([-N/2, N/2,min(np.unwrap(np.angle(X))),max(np.unwrap(np.angle(X)))])
plt.show()