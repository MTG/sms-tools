import numpy as np
import matplotlib.pyplot as plt

N = 64
k0 = 10.5

plt.figure(1)
X = np.array([])
x = np.exp(1j*2*np.pi*k0/(N+1)*np.arange(-N/2, N/2+1))

plt.subplot(4,2,1)
plt.title ('Complex sinewave: x1')
plt.plot(np.arange(-N/2, N/2+1), np.real(x))
plt.plot(np.arange(-N/2, N/2+1), np.imag(x))
plt.axis([-N/2,N/2,-1,1])
for k in range(-N/2, N/2+1):
	s = np.exp(1j*2*np.pi*k/(N+1)*np.arange(-N/2, N/2+1))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(4,2,3)
plt.title ('Complex spectrum: X1')
plt.plot(np.arange(-N/2, N/2+1), np.real(X))
plt.plot(np.arange(-N/2, N/2+1), np.imag(X))
plt.axis([-N/2,N/2,-15,50])

plt.subplot(4,2,5)
plt.title ('Magnitude spectrum: abs(X1)')
plt.plot(np.arange(-N/2, N/2+1), abs(X))
plt.axis([-N/2,N/2,0,50])

plt.subplot(4,2,7)
plt.title ('Phase spectrum: angle(X1)')
plt.plot(np.arange(-N/2, N/2+1), np.angle(X))
plt.axis([-N/2, N/2,-np.pi,np.pi])

X = np.array([])
x = np.cos(2*np.pi*k0/(N+1)*np.arange(-N/2, N/2+1))

plt.subplot(4,2,2)
plt.title ('Real sinewave: x2')
plt.plot(np.arange(-N/2, N/2+1), x)
plt.axis([-N/2,N/2,-1,1])
for k in range(-N/2, N/2+1):
	s = np.exp(1j*2*np.pi*k/(N+1)*np.arange(-N/2, N/2+1))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(4,2,4)
plt.title ('Complex spectrum: X2')
plt.plot(np.arange(-N/2, N/2+1), np.real(X))
plt.plot(np.arange(-N/2, N/2+1), np.imag(X))
plt.axis([-N/2,N/2,-15,50])

plt.subplot(4,2,6)
plt.title ('Magnitude spectrum: abs(X2)')
plt.plot(np.arange(-N/2, N/2+1), abs(X))
plt.axis([-N/2,N/2,0,50])

plt.subplot(4,2,8)
plt.title ('Phase spectrum: angle(X2)')
plt.plot(np.arange(-N/2, N/2+1), np.angle(X))
plt.axis([-N/2, N/2,-np.pi,np.pi])

plt.show()