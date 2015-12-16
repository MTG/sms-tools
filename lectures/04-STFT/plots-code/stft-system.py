import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import stft as STFT
import utilFunctions as UF
import matplotlib.pyplot as plt
from scipy.signal import hamming


(fs, x) = UF.wavread('../../../sounds/piano.wav')
w = np.hamming(1024)
N = 1024
H = 512
mX, pX = STFT.stftAnal(x, w, N, H)
y = STFT.stftSynth(mX, pX, w.size, H)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(411)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.title('x (piano.wav)')
plt.axis([0,x.size/float(fs),min(x),max(x)])

plt.subplot(412)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(mX[0,:].size)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
plt.title('mX, M=1024, N=1024, H=512')
plt.autoscale(tight=True)

plt.subplot(413)
numFrames = int(pX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(pX[0,:].size)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.diff(np.transpose(pX),axis=0))
plt.title('pX derivative, M=1024, N=1024, H=512')
plt.autoscale(tight=True)

plt.subplot(414)
plt.plot(np.arange(y.size)/float(fs), y,'b')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('y')

plt.tight_layout()
plt.savefig('stft-system.png')
UF.wavwrite(y, fs, 'piano-stft.wav')
plt.show()
  
  
  
