import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import stft as STFT
import utilFunctions as UF
import matplotlib.pyplot as plt
from scipy.signal import hamming
from scipy.fftpack import fft
import math

(fs, x) = UF.wavread('../../../sounds/piano.wav')

plt.figure(1, figsize=(9.5, 6))

w = np.hamming(256)
N = 256
H = 128
mX1, pX1 = STFT.stftAnal(x, w, N, H)
plt.subplot(211)
numFrames = int(mX1[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(mX1[0,:].size)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX1))
plt.title('mX (piano.wav), M=256, N=256, H=128')
plt.autoscale(tight=True)

w = np.hamming(1024)
N = 1024
H = 128
mX2, pX2 = STFT.stftAnal(x, w, N, H)

plt.subplot(212)
numFrames = int(mX2[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(mX2[0,:].size)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX2))
plt.title('mX (piano.wav), M=1024, N=1024, H=128')
plt.autoscale(tight=True)

plt.tight_layout()
plt.savefig('time-freq-compromise.png')
plt.show()

