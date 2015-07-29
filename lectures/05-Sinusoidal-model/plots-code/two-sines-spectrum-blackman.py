import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
import dftModel as DF
(fs, x) = UF.wavread('../../../sounds/sine-440-490.wav')
w = np.blackman(5291)
N = 16384
pin = .11*fs
x1 = x[pin:pin+w.size]
mX, pX = DF.dftAnal(x1, w, N)

plt.figure(1, figsize=(9.5, 5))
plt.subplot(311)
plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b', lw=1.5)
plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1)-.01, max(x1)+.01])
plt.title('x (sine-440-490.wav), M=5291')

plt.subplot(3,1,2)
plt.plot(fs*np.arange(mX.size)/float(N), mX, 'r', lw=1.5)
plt.axis([100,900,-85,max(mX)+1])
plt.title ('mX, N=16384, blackman window')

plt.subplot(3,1,3)
plt.plot(fs*np.arange(mX.size)/float(N), pX, 'c', lw=1.5)
plt.axis([100,900,-1,18])
plt.title ('pX')

plt.tight_layout()
plt.savefig('two-sines-spectrum-blackman.png')
plt.show()
