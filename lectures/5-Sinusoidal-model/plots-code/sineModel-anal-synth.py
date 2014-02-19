import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import stftAnal, sineModelAnal, sineModelSynth
import smsWavplayer as wp


(fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/bendir.wav'))
w = np.hamming(1001)
N = 2048
t = -70
Ns = 1024
H = Ns/4
ploc, pmag, pphase = sineModelAnal.sineModelAnal(x, fs, w, N, H, t)
y = sineModelSynth.sineModelSynth(ploc, pmag, pphase, N, Ns, H)
numFrames = int(ploc[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)

plt.figure(1)

plt.subplot(3,1,1)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.axis([0,x.size/float(fs),min(x),max(x)])
plt.title('x = wavread("bendir.wav")')                        

plt.subplot(3,1,2)
yploc = ploc
yploc[ploc==0] = np.nan
plt.plot(frmTime, yploc, '.', color='k', alpha=yploc/max(yploc))
plt.axis([0,y.size/float(fs),0,400])
plt.title('spectral peaks')

plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y, 'b')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('y')    

plt.show()
