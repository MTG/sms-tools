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


(fs, x) = WIO.wavread('../../../sounds/vignesh.wav')
w = np.blackman(1201)
N = 2048
t = -90
minf0 = 130
maxf0 = 300
f0et = 7
maxnpeaksTwm = 4
H = 128

mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
f0 = f0Twm.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et, maxnpeaksTwm)
maxplotfreq = 2000.0
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.autoscale(tight=True)
  
plt.plot(frmTime, f0, linewidth=2, color='k')
plt.autoscale(tight=True)
plt.title('f0 on spectrogram')
plt.show()

