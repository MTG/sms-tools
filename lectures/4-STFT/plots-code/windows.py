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
N = 512
pin = 5000
w = np.ones(501)
hM1 = int(math.floor((w.size+1)/2)) 
hM2 = int(math.floor(w.size/2))  
x1 = x[pin-hM1:pin+hM2]


plt.figure(1)
plt.subplot(4,1,1)
plt.plot(np.arange(-hM1, hM2), x1)
plt.axis([-hM1, hM2, min(x1), max(x1)])
plt.title('x = wavread(oboe-A4.wav)')

mX, pX = dftAnal.dftAnal(x1, w, N)
mX = mX - max(mX)

plt.subplot(4,1,2)
plt.plot(np.arange(N/2), mX, 'r')
plt.axis([0,N/4,-70,0])
plt.title ('magnitude spectrum with Reactangular window')

w = np.hamming(501)
mX, pX = dftAnal.dftAnal(x1, w, N)
mX = mX - max(mX)

plt.subplot(4,1,3)
plt.plot(np.arange(N/2), mX, 'r')
plt.axis([0,N/4,-70,0])
plt.title ('magnitude spectrum with Hamming window')

w = np.blackman(501)
mX, pX = dftAnal.dftAnal(x1, w, N)
mX = mX - max(mX)

plt.subplot(4,1,4)
plt.plot(np.arange(N/2), mX, 'r')
plt.axis([0,N/4,-70,0])
plt.title ('magnitude spectrum with Blackman window')
plt.show()
