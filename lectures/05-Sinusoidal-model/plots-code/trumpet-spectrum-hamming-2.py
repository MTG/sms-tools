import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF
import dftModel as DF
(fs, x) = UF.wavread('../../../sounds/trumpet-A4.wav')
w = np.hamming(801)
N = 2048
pin = 5000
x1 = x[pin:pin+w.size]
mX, pX = DF.dftAnal(x1, w, N)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(311)
plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b',lw=1.5)
plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1), max(x1)])
plt.title('x (trumpet-A4.wav), M=801')

plt.subplot(3,1,2)
plt.plot(fs*np.arange(mX.size)/float(N), mX, 'r', lw=1.5)
plt.axis([0,8000,-90,max(mX)+1])
plt.title ('mX; Hamming window, N=2048')

plt.subplot(3,1,3)
plt.plot(fs*np.arange(pX.size)/float(N), pX, 'c', lw=1.5)
plt.axis([0,8000,-14,7])
plt.title ('pX; Hamming window, N=2048')

plt.tight_layout()
plt.savefig('trumpet-spectrum-hamming-2.png')
plt.show()
