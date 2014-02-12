import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))

import stftAnal, stftSynth
import smsWavplayer as wp
import matplotlib.pyplot as plt
from scipy.signal import hamming


(fs, x) = wp.wavread('../../../sounds/piano.wav')
w = np.hamming(1024)
N = 1024
H = 512
mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
y = stftSynth.stftSynth(mX, pX, w.size, H)

plt.figure(1)
plt.subplot(411)
plt.plot(np.arange(x.size)/float(fs), x,'b')
plt.title('input sound x=wavread(piano.wav)')
plt.axis([0,x.size/float(fs),min(x),max(x)])

plt.subplot(412)
numFrames = int(mX[1,:].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, mX)
plt.title('magnitude spectrogram; M=1024, N=1024, H=512')
plt.autoscale(tight=True)

plt.subplot(413)
numFrames = int(pX[1,:].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.diff(pX,axis=0))
plt.title('phase spectrogram (derivative); M=1024, N=1024, H=512')
plt.autoscale(tight=True)

plt.subplot(414)
plt.plot(np.arange(y.size)/float(fs), y,'b')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('output sound y')

plt.show()
  
  
  
