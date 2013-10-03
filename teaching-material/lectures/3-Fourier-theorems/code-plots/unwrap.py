import numpy as np
import matplotlib.pyplot as plt

N = 64
k0 = 10.5

plt.figure(1)
X = np.array([])
x = np.cos(2*np.pi*k0/(N+1)*np.arange(-N/2, N/2+1))

plt.subplot(411)
plt.title('x')
plt.plot(np.arange(-N/2, N/2+1), x)
plt.axis([-N/2,N/2,-1,1])
for k in range(-N/2, N/2+1):
	s = np.exp(1j*2*np.pi*k/(N+1)*np.arange(-N/2, N/2+1))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(412)
plt.title('mag spectrum: abs(X)')
plt.plot(np.arange(-N/2, N/2+1), abs(X))
plt.axis([-N/2,N/2,0,25])

plt.subplot(413)
plt.title('phase spectrum: angle(X)')
plt.plot(np.arange(-N/2, N/2+1), np.angle(X))
plt.axis([-N/2, N/2,-np.pi,np.pi])

pX = np.unwrap(np.angle(X))

plt.subplot(414)
plt.title('unwrapped phase spectrum: unwrap(angle(X))')
plt.plot(np.arange(-N/2, N/2+1), pX)
plt.axis([-N/2, N/2,min(pX), max(pX)])
plt.show()