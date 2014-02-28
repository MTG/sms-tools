import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import time
import math
import sys, os, functools
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))

import waveIO as WIO
import peakProcessing as PP
import harmonicDetection as HD
import errorHandler as EH
import stftAnal, dftAnal, f0Twm

(fs, x) = WIO.wavread('../../sounds/vignesh.wav')
w = np.blackman(1201)
N = 2048
t = -90
nH = 100
Ns = 512
H = Ns/4
winLen = 1024

mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
maxplotfreq = 1000.0
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.autoscale(tight=True)

pitch = ess.PredominantMelody(hopSize = H, frameSize = winLen)(x) 
pitch = pitch[0]                                                                # Select the pitch track, not the salience
tPitch = np.array(range(0,len(pitch)))*np.float(H)/fs                    
plt.plot(tPitch,pitch,'b', linewidth=1.0)

t = -90
minf0 = 130
maxf0 = 300
f0et = 7
maxnpeaksTwm = 4
f0 = f0Twm.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et, maxnpeaksTwm)
tf0 = np.array(range(0,len(f0)))*np.float(H)/fs
plt.plot(tf0, f0,'k', linewidth=1.0)
  
plt.show()


    