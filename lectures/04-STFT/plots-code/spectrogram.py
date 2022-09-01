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
w = np.hamming(1001)
N = 1024
H = 256
mX, pX = STFT.stftAnal(x, w, N, H)

plt.figure(1, figsize=(9.5, 6))

plt.subplot(211)
numFrames = int(mX[:,0].size) + 1
# Size of X and Y must be 1 larger than the size of mX for flat shading
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2+1+1)*float(fs)/N
binFreq2 = np.arange(mX[:,0].size)*float(fs)/N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX), shading = 'flat')
plt.title('mX (piano.wav), Hamming window, M=1001, N=1024, H=256')
plt.autoscale(tight=True)

plt.subplot(212)
# Size of X must be 1 larger than the size of np.diff(pX) (which has the Y axis size
# reduced by 1) for flat shading
numFrames = int(pX[:,0].size) + 1
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(pX),axis=0), shading = 'flat')
plt.title('pX derivative (piano.wav), Hamming window, M=1001, N=1024, H=256')
plt.autoscale(tight=True)

plt.tight_layout()
plt.savefig('spectrogram.png')
plt.show()
