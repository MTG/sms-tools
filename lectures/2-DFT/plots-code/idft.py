import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))

import dftAnal, dftSynth
import smsWavplayer as wp
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft
import math

k0 = 8.5
N = 64
w = np.ones(N)
x = np.cos(2*np.pi*k0/N*np.arange(-N/2,N/2))
mX, pX = dftAnal.dftAnal(x, w, N)
y = dftSynth.dftSynth(mX, pX, N)

plt.figure(1)
plt.subplot(311)
plt.title('positive freq. magnitude spectrum in dB: mX')
plt.plot(np.arange(0, N/2), mX, 'r')
plt.axis([0,N/2,min(mX),max(mX)+1])

plt.subplot(312)
plt.title('positive freq. phase spectrum: pX')
plt.plot(np.arange(0, N/2), pX, 'c')
plt.axis([0,N/2,-np.pi,np.pi])

plt.subplot(313)
plt.title('inverse spectrum: IDFT(X)')
plt.plot(np.arange(-N/2, N/2), y,'b')
plt.axis([-N/2,N/2-1,min(y), max(y)])
plt.show()
