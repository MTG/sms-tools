import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import sys, os
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import waveIO as WIO
import stftAnal, dftAnal, f0Yin, f0Twm

(fs, x) = WIO.wavread('../../../sounds/carnatic-2.wav')
N = 2048
minf0 = 130
maxf0 = 300
H = 256
  
w = hamming(801)
mX, pX = stftAnal.stftAnal(x, fs, w, N, H)
maxplotfreq = 600.0
frmTime = H*np.arange(mX[:,0].size)/float(fs)
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.autoscale(tight=True)
  
f0Y = f0Yin.f0Yin(x, N, H, minf0, maxf0)
frmTime = H*np.arange(f0Y.size)/float(fs)  
plt.plot(frmTime, f0Y, linewidth=1, color='k')
plt.autoscale(tight=True)

pitch = ess.PredominantMelody(hopSize = H, frameSize = N, guessUnvoiced=True)(x)     
f0M= pitch[0]                                                             
frmTime = H*np.arange(f0M.size-4)/float(fs) 
plt.plot(frmTime, f0M[4:],'b', linewidth=1.0)

w = np.blackman(2048)
t = -90
f0et = 7
maxnpeaksTwm = 4
f0T = f0Twm.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et, maxnpeaksTwm)
frmTime = H*np.arange(f0T.size)/float(fs) 
plt.plot(frmTime, f0T,'c', linewidth=1.0)
plt.autoscale(tight=True)

plt.show()
