import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import math
import sys, os, functools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import waveIO as WIO
import peakProcessing as PP
import harmonicDetection as HD
import errorHandler as EH
import stftAnal, dftAnal, f0Twm

try:
  import genSpecSines_C as GS
  import twm_C as TWM
except ImportError:
  import genSpecSines as GS
  import twm as TWM
  EH.printWarning(1)


(fs, x) = WIO.wavread('../../../sounds/piano.wav')
w = np.blackman(1501)
N = 2048
t = -90
minf0 = 100
maxf0 = 300
f0et = 1
maxnpeaksTwm = 4
H = 128
x1 = x[1.5*fs:1.8*fs]

mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
f0 = f0Twm.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et, maxnpeaksTwm)
f0[f0==0] = np.nan
maxplotfreq = 800.0
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.autoscale(tight=True)
  
plt.plot(frmTime, f0, linewidth=2, color='k')
plt.autoscale(tight=True)
plt.title('f0 on spectrogram; TWM; (piano.wav)')
plt.show()

