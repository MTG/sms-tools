import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import waveIO as WIO
import dftAnal as DF
(fs, x) = WIO.wavread('../../../sounds/sine-440+490.wav')
w = np.hamming(3529)
N = 16384
pin = 4850
x1 = x[pin:pin+w.size]
mX, pX = DF.dftAnal(x1, w, N)

plt.figure(1)
plt.subplot(311)
plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b')
plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1), max(x1)])
plt.title('x=wavread(sine-440+490.wav); M=3529')

plt.subplot(3,1,2)
plt.plot(fs*np.arange(0,N/2)/float(N), mX, 'r')
plt.axis([100,830,-80,max(mX)])
plt.title ('mX; N=16384')

plt.subplot(3,1,3)
plt.plot(fs*np.arange(0,N/2)/float(N), pX, 'c')
plt.axis([100,830,60,150])
plt.title ('pX')

plt.show()
