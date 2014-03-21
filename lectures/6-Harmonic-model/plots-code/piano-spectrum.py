import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math
from scipy.signal import blackman

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import waveIO as WIO
import dftAnal as DF

(fs, x) = WIO.wavread('../../../sounds/piano.wav')
pin = .3*fs
w = np.blackman(1001)
N = 1024
hM1 = int(math.floor((w.size+1)/2)) 
hM2 = int(math.floor(w.size/2))  
x1 = x[pin-hM1:pin+hM2]
mX, pX = DF.dftAnal(x1, w, N)

plt.figure(1)
plt.subplot(311)
plt.plot(np.arange(-hM1, hM2)/float(fs), x1, lw=1.5)
plt.axis([-hM1/float(fs), hM2/float(fs), min(x1), max(x1)])
plt.ylabel('amplitude')
plt.title('x (piano.wav)')

plt.subplot(3,1,2)
plt.plot(fs*np.arange(N/2)/float(N), mX, 'r', lw=1.5)
plt.axis([0,fs/2,-110,max(mX)])
plt.title ('mX, magnitude spectrum')
plt.ylabel('amplitude (dB)')

plt.subplot(3,1,3)
plt.plot(fs*np.arange(N/2)/float(N), pX, 'c', lw=1.5)
plt.axis([0,fs/2,min(pX),max(pX)])
plt.title ('pX, phase spectrum')
plt.ylabel('phase (radians)')


plt.show()