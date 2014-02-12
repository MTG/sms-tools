import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
from scipy import signal

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import dftAnal as DF

N = 128
k0 = 1
x = np.cos(2*np.pi*k0/N*np.arange(-N/2, N/2+1))
plt.figure(1)

mX, pX = DF.dftAnal(x[0:N], np.ones(N), N)

plt.subplot(3,2,1)
plt.title ('x1=x[n]')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), x[0:N])
plt.axis([-(N-1)/2.0,(N+1)/2-1,min(x),max(x)])

plt.subplot(3,2,3)
plt.title ('magnitude spectrum: abs(X1)')
plt.plot(np.arange(N/2), mX, 'r')
plt.axis([0,N/2-1,-45,max(mX)])

plt.subplot(3,2,5)
plt.title ('phase spectrum: angle(X1)')
plt.plot(np.arange(N/2), pX, 'c')
plt.axis([0,N/2-1,min(pX),max(pX)])

x = np.sin(2*np.pi*k0/N*np.arange(-N/2, N/2+1))
mX, pX = DF.dftAnal(x[0:N], np.ones(N), N)

plt.subplot(3,2,2)
plt.title ('x2=x[n-2]')
plt.plot(np.arange(-(N-1)/2.0,(N+1)/2), x[0:N])
plt.axis([-(N-1)/2.0,(N+1)/2-1,min(x),max(x)])

plt.subplot(3,2,4)
plt.title ('magnitude spectrum: abs(X2)')
plt.plot(np.arange(N/2), mX, 'r')
plt.axis([0,N/2-1,-45,max(mX)])

plt.subplot(3,2,6)
plt.title ('phase spectrum: angle(X2)')
plt.plot(np.arange(N/2), pX, 'c')
plt.axis([0,N/2-1,min(pX),max(pX)])

plt.show()
