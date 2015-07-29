import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftModel as DF
import utilFunctions as UF
from scipy.fftpack import fft, ifft
import math

(fs, x) = UF.wavread('../../../sounds/oboe-A4.wav')
N = 512
pin = 5000
w = np.ones(501)
hM1 = int(math.floor((w.size+1)/2)) 
hM2 = int(math.floor(w.size/2))  
x1 = x[pin-hM1:pin+hM2]


plt.figure(1, figsize=(9.5, 7))
plt.subplot(4,1,1)
plt.plot(np.arange(-hM1, hM2), x1, lw=1.5)
plt.axis([-hM1, hM2, min(x1), max(x1)])
plt.title('x (oboe-A4.wav)')

mX, pX = DF.dftAnal(x1, w, N)
mX = mX - max(mX)

plt.subplot(4,1,2)
plt.plot(np.arange(mX.size), mX, 'r', lw=1.5)
plt.axis([0,N/4,-70,0])
plt.title ('mX (rectangular window)')

w = np.hamming(501)
mX, pX = DF.dftAnal(x1, w, N)
mX = mX - max(mX)

plt.subplot(4,1,3)
plt.plot(np.arange(mX.size), mX, 'r', lw=1.5)
plt.axis([0,N/4,-70,0])
plt.title ('mX (hamming window)')

w = np.blackman(501)
mX, pX = DF.dftAnal(x1, w, N)
mX = mX - max(mX)

plt.subplot(4,1,4)
plt.plot(np.arange(mX.size), mX, 'r', lw=1.5)
plt.axis([0,N/4,-70,0])
plt.title ('mX (blackman window)')

plt.tight_layout()
plt.savefig('windows.png')
plt.show()
