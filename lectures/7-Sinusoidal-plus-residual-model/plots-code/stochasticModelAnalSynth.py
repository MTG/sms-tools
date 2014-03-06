import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, resample
from scipy.fftpack import fft, ifft
import time
import sys, os

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import waveIO as WIO
import stochasticModelAnal, stochasticModel,stftAnal
  
(fs, x) = WIO.wavread('../../../sounds/ocean.wav')
w = np.hamming(512)
N = 512
H = 256
stocf = .1
mXenv = stochasticModelAnal.stochasticModelAnal(x, w, N, H, stocf)
y = stochasticModel.stochasticModel(x, w, N, H, stocf)
mX, pX = stftAnal.stftAnal(x, fs, w, N, H)


plt.figure(1)
plt.subplot(411)
plt.plot(np.arange(x.size)/float(fs), x,'b')
plt.title('input sound x=wavread(ocean.wav)')
plt.axis([0,x.size/float(fs),min(x),max(x)])

plt.subplot(412)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
plt.title('magnitude spectrogram; M=512, N=512, H=256')
plt.autoscale(tight=True)

plt.subplot(413)
numFrames = int(mXenv[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(stocf*N/2)*float(fs)/(stocf*N)                       
plt.pcolormesh(frmTime, binFreq, np.transpose(mXenv))
plt.title('stochastic approximation; stocf=.1')
plt.autoscale(tight=True)

plt.subplot(414)
plt.plot(np.arange(x.size)/float(fs), y,'b')
plt.title('y')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.show()
