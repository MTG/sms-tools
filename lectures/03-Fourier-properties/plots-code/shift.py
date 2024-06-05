import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import sawtooth
from smstools.models import dftModel as DF

N = 32
n = np.arange(-N//2+1, N//2+1)
x1 = sawtooth(2*np.pi*n/float(N))
x2 = np.roll(x1,2)
mX1, pX1 = DF.dftAnal(x1, np.ones(N), N)
mX2, pX2 = DF.dftAnal(x2, np.ones(N), N)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(321)
plt.title('x1=x[n]')
plt.plot(n, x1, 'b-x', lw=1.5)
plt.axis([-N//2+3, N//2, -1, 1])

plt.subplot(322)
plt.title('x2=x[n-2]')
plt.plot(n, x2, 'b-x', lw=1.5)
plt.axis([-N//2+3, N//2, -1, 1])

plt.subplot(323)
plt.title('mX1')
plt.plot(np.arange(0, mX1.size, 1.0), mX1, 'r-x', lw=1.5)
plt.axis([0,mX1.size-1,min(mX1),max(mX1)])

plt.subplot(324)
plt.title('mX2')
plt.plot(np.arange(0, mX2.size, 1.0), mX2, 'r-x', lw=1.5)
plt.axis([0,mX2.size-1,min(mX2),max(mX2)])

plt.subplot(325)
plt.title('pX1')
plt.plot(np.arange(0, pX1.size, 1.0), pX1, 'c-x', lw=1.5)
plt.axis([0,pX1.size-1,min(pX1),max(pX1)])

plt.subplot(326)
plt.title('pX2')
plt.plot(np.arange(0, pX2.size, 1.0), pX2, 'c-x', lw=1.5)
plt.axis([0,pX2.size-1,min(pX2),max(pX2)])

plt.tight_layout()
plt.savefig('shift.png')
plt.show()
