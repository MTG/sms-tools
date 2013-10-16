import numpy as np
import matplotlib.pyplot as plt

plt.figure(1)
N = 64
k0 = 7.5
X = np.array([])
y = np.array([])
plt.figure(1)
x = np.cos(2*np.pi*k0/N*np.arange(-N/2, N/2))
for k in np.arange(-N/2, N/2):
	s = np.exp((-1j*2*np.pi*k/N)*np.arange(-N/2, N/2))
	X = np.append(X, sum(x*np.conjugate(s)))


plt.subplot(411)
plt.title('magnitude spectrum: abs(X)')
plt.plot(np.arange(-N/2, N/2), abs(X), 'ro')
plt.axis([-N/2,N/2-1,0,N/2])

plt.subplot(412)
plt.title('phase spectrum: angle(X)')
plt.plot(np.arange(-N/2, N/2), np.angle(X), 'g')
plt.axis([-N/2,N/2-1,-np.pi,np.pi])

plt.subplot(413)
plt.title('complex spectrum: X')
plt.plot(np.arange(-N/2, N/2), np.real(X))
plt.plot(np.arange(-N/2, N/2), np.imag(X))
plt.axis([-N/2,N/2-1,-N/2,N/2])

for k in np.arange(-N/2, N/2):
	s = np.exp((1j*2*np.pi*k/N)*np.arange(-N/2, N/2))
	y = np.append(y, sum(X*s)/N)
plt.subplot(414)
plt.title('inverse spectrum: IDFT(X)')
plt.plot(np.arange(-N/2, N/2), y)
plt.axis([-N/2,N/2-1,-1,1])
plt.show()
