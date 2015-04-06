import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF
import dftModel as DF
(fs, x) = UF.wavread('../../../sounds/oboe-A4.wav')
w = np.hamming(401)
N = 1024
pin = 5000
x1 = x[pin:pin+w.size]
mX, pX = DF.dftAnal(x1, w, N)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(311)
plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b', lw=1.5)
plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1), max(x1)])
plt.title('x (oboe-A4.wav), M=401')

plt.subplot(3,1,2)
plt.plot(fs*np.arange(mX.size)/float(N), mX, 'r', lw=1.5)
plt.axis([0,8000,-80,max(mX)])
plt.title ('mX; Hamming window, N=1024')

plt.subplot(3,1,3)
plt.plot(fs*np.arange(mX.size)/float(N), pX, 'c', lw=1.5)
plt.axis([0,8000,-12,15])
plt.title ('pX; Hamming window, N=1024')

plt.tight_layout()
plt.savefig('oboe-spectrum-hamming-1.png')
plt.show()
