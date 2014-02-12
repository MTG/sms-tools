import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import smsWavplayer as wp
import dftAnal as DF
(fs, x) = wp.wavread('../../../sounds/oboe-A4.wav')
w = np.hamming(401)
N = 1024
pin = 5000
x1 = x[pin:pin+w.size]
mX, pX = DF.dftAnal(x1, w, N)

plt.figure(1)
plt.subplot(311)
plt.plot(np.arange(pin, pin+w.size)/float(fs), x1, 'b')
plt.axis([pin/float(fs), (pin+w.size)/float(fs), min(x1), max(x1)])
plt.title('input signal: x=wavread(oboe-A4.wav)')

plt.subplot(3,1,2)
plt.plot(fs*np.arange(0,N/2)/float(N), mX, 'r')
plt.axis([0,8000,-80,max(mX)])
plt.title ('mX; Hamming window, M=401, N=1024')

plt.subplot(3,1,3)
plt.plot(fs*np.arange(0,N/2)/float(N), pX, 'c')
plt.axis([0,8000,-10,20])
plt.title ('phase spectrum: pX')

plt.show()
