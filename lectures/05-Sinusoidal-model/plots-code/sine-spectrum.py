import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftModel as DFT
import utilFunctions as UF

(fs, x) = UF.wavread('../../../sounds/sine-440.wav')
M = 400
x1 = x[2000:2000+M]
N = 2048
hM = int(M/2.0)
w = np.hamming(M)
mX, pX = DFT.dftAnal(x1, w, N)
freqaxis = fs*np.arange(0,mX.size)/float(N)
taxis = np.arange(N)/float(fs) 

plt.figure(1, figsize=(9.5, 7))

plt.subplot(3,1,1)
plt.plot(np.arange(M)/float(fs), x1, 'b', lw=1.5)
plt.axis([0,(M-1)/float(fs),min(x1)-.1,max(x1)+.1])
plt.title ('x (sine-440.wav)')

plt.subplot(3,1,2)
plt.plot(freqaxis, mX, 'r', lw=1.5)
plt.axis([0,fs/10,-80,max(mX)+1])
plt.title ('mX')

plt.subplot(3,1,3)
plt.plot(freqaxis, pX, 'c', lw=1.5)
plt.axis([0,fs/10,-60,max(pX)])
plt.title ('pX')


plt.tight_layout()
plt.savefig('sine-spectrum.png')
plt.show()
