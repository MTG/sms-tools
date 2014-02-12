import matplotlib.pyplot as plt
import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftAnal, dftSynth
import smsWavplayer as wp
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft
import math

(fs, x) = wp.wavread('../../../sounds/oboe-A4.wav')
w = np.hanning(501)
N = 512
pin = 5000
hM1 = int(math.floor((w.size+1)/2)) 
hM2 = int(math.floor(w.size/2))  
x1 = x[pin-hM1:pin+hM2]
mX, pX = dftAnal.dftAnal(x1, w, N)
y = dftSynth.dftSynth(mX, pX, w.size)*sum(w)

plt.figure(1)
plt.subplot(4,1,1)
plt.plot(np.arange(-hM1, hM2), x1*w)
plt.axis([-hM1, hM2, min(x1), max(x1)])
plt.ylabel('amplitude')
plt.title('input signal: x = wavread(oboe-A4.wav)')

plt.subplot(4,1,2)
plt.plot(np.arange(N/2), mX, 'r')
plt.axis([0,N/2,min(mX),max(mX)])
plt.title ('magnitude spectrum: mX')
plt.ylabel('amplitude (dB)')
plt.ylabel('frequency samples')

plt.subplot(4,1,3)
plt.plot(np.arange(N/2), pX, 'c')
plt.axis([0,N/2,min(pX),max(pX)])
plt.title ('phase spectrum: pX')
plt.ylabel('phase (radians)')
plt.ylabel('frequency samples')

plt.subplot(4,1,4)
plt.plot(np.arange(-hM1, hM2), y)
plt.axis([-hM1, hM2, min(y), max(y)])
plt.ylabel('amplitude')
plt.title('output signal: y')
plt.show()
