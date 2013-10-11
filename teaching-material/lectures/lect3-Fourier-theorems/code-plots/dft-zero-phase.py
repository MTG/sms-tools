import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftshift

N = 41
nsines = 20

plt.figure(1)
X = np.array([])
x = np.zeros(N)
for k in range(1,nsines):
	x = x + (1.0/k)*np.sin(2*np.pi*k/N*np.arange(-(N-1)/2.0, (N+1)/2.0))
plt.subplot(3,2,1)
plt.title ('x[n]')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), x)
plt.axis([-(N-1)/2.0,(N+1)/2,min(x),max(x)])

for k in range(N):
	s = np.exp(1j*2*np.pi*k/N*np.arange(-(N-1)/2.0, (N+1)/2.0))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(3,2,3)
plt.title ('magnitude spectrum: abs(X)')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), abs(fftshift(X))/N, 'ro')
plt.axis([-(N-1)/2.0,1+(N-1)/2.0,0,max(abs(X)/N)])

plt.subplot(3,2,5)
plt.title ('phase spectrum: angle(X)')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), np.angle(fftshift(X)), 'bo')
plt.axis([-(N-1)/2.0,1+(N-1)/2.0,min(np.angle(fftshift(X))),max(np.angle(fftshift(X)))])

X = np.array([])
x = np.zeros(N)
for k in range(1,nsines):
	x = x + (1.0/k)*np.sin(2*np.pi*k/N*np.arange(-(N-1)/2.0-1, (N+1)/2.0-1))
plt.subplot(3,2,2)
plt.title ('x1[n] = x[n-1]')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), x)
plt.axis([-(N-1)/2.0,(N+1)/2,min(x),max(x)])

for k in range(N):
	s = np.exp(1j*2*np.pi*k/N*np.arange(-(N-1)/2.0, (N+1)/2.0))
	X = np.append(X, sum(x*np.conjugate(s)))

plt.subplot(3,2,4)
plt.title ('magnitude spectrum: abs(X1)')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), abs(fftshift(X))/N, 'ro')
plt.axis([-(N-1)/2.0,1+(N-1)/2.0,0,max(abs(X)/N)])

plt.subplot(3,2,6)
plt.title ('phase spectrum: angle(X1)')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), np.unwrap(np.angle(fftshift(X))), 'bo')
plt.axis([-(N-1)/2.0,1+(N-1)/2.0,min(np.unwrap(np.angle(fftshift(X)))),max(np.unwrap(np.angle(fftshift(X))))])

plt.show()