import numpy as np
import time, os, sys
from scipy.signal import hamming, resample
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/transformations/'))

import dftAnal as DFT
import dftSynth as IDFT
import waveIO as WIO
import stftMorph as STFTM
import stochasticModelAnal as STOC
import math
import stftAnal as STFT

(fs, x1) = WIO.wavread('../../../sounds/orchestra.wav')
(fs, x2) = WIO.wavread('../../../sounds/speech.wav')
w1 = np.hamming(1024)
N1 = 1024
H1 = 256
w2 = np.hamming(1024)
N2 = 1024
smoothf = .2
balancef = 0.5
y = STFTM.stftMorph(x1, x2, fs, w1, N1, w2, N2, H1, smoothf, balancef)
L = int(x1.size/H1)
H2 = int(x2.size/L)
mX2 = STOC.stochasticModelAnal(x2,H2,smoothf)
mX,pX = STFT.stftAnal(x1, fs, w1, N1, H1)
mY,pY = STFT.stftAnal(y, fs, w1, N1, H1)
maxplotfreq = 10000.0

plt.figure(1)
plt.subplot(311)
numFrames = int(mX[:,0].size)
frmTime = H1*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N1*maxplotfreq/fs)/N1                       
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N1*maxplotfreq/fs+1])) 
plt.title('mX1 (orchestra.wav)')
plt.autoscale(tight=True)

plt.subplot(312)
numFrames = int(mX2[:,0].size)
frmTime = H2*np.arange(numFrames)/float(fs)  
                 
N = 2*mX2[0,:].size         
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                       
plt.pcolormesh(frmTime, binFreq, np.transpose(mX2[:,:N*maxplotfreq/fs+1]))
plt.title('mX2 (speech.wav)')
plt.autoscale(tight=True)

plt.subplot(313)
numFrames = int(mY[:,0].size)
frmTime = H1*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N1*maxplotfreq/fs)/N1                       
plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:,:N1*maxplotfreq/fs+1])) 
plt.title('mY')
plt.autoscale(tight=True)

plt.show()