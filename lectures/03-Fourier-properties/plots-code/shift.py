import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.signal import sawtooth
sys.path.append('../../../software/models/')
import dftModel as DF

N = 128
x1 = sawtooth(2*np.pi*np.arange(-N/2,N/2)/float(N))
x2 = sawtooth(2*np.pi*np.arange(-N/2-2,N/2-2)/float(N))
mX1, pX1 = DF.dftAnal(x1, np.ones(N), N)
mX2, pX2 = DF.dftAnal(x2, np.ones(N), N)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(321)
plt.title('x1=x[n]')
plt.plot(np.arange(-N/2, N/2, 1.0), x1, lw=1.5)
plt.axis([-N/2, N/2, -1, 1])

plt.subplot(322)
plt.title('x2=x[n-2]')
plt.plot(np.arange(-N/2, N/2, 1.0), x2, lw=1.5)
plt.axis([-N/2, N/2, -1, 1])

plt.subplot(323)
plt.title('mX1')
plt.plot(np.arange(0, mX1.size, 1.0), mX1, 'r', lw=1.5)
plt.axis([0,mX1.size,min(mX1),max(mX1)])      

plt.subplot(324)
plt.title('mX2')
plt.plot(np.arange(0, mX2.size, 1.0), mX2, 'r', lw=1.5)
plt.axis([0,mX2.size,min(mX2),max(mX2)])  

plt.subplot(325)
plt.title('pX1')
plt.plot(np.arange(0, pX1.size, 1.0), pX1, 'c', lw=1.5)
plt.axis([0,pX1.size,min(pX1),max(pX2)])  

plt.subplot(326)
plt.title('pX2')
plt.plot(np.arange(0, pX2.size, 1.0), pX2, 'c', lw=1.5)
plt.axis([0,pX2.size,min(pX2),max(pX2)]) 

plt.tight_layout()
plt.savefig('shift.png')
plt.show()
