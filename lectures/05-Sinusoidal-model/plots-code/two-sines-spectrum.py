import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
import dftModel as DF
(fs, x) = UF.wavread('../../../sounds/sine-440-490.wav')
w = np.hamming(3501)
N = 4096
pin = 5000
x1 = x[pin:pin+w.size]
mX, pX = DF.dftAnal(x1, w, N)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(311)
plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b', lw=1.5)
plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1), max(x1)])
plt.title('x (sine-440-490.wav)')

plt.subplot(3,1,2)
plt.plot(fs*np.arange(mX.size)/float(N), mX, 'r', lw=1.5)
plt.axis([0,1200,-80,max(mX)+1])
plt.title ('mX')

plt.subplot(3,1,3)
plt.plot(fs*np.arange(mX.size)/float(N), pX, 'c', lw=1.5)
plt.axis([0,1200,min(pX),10])
plt.title ('pX')

plt.tight_layout()
plt.savefig('two-sines-spectrum.png')
plt.show()
