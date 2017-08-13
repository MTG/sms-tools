import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import sys, os, functools, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import sineModel as SM
import utilFunctions as UF

(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/bendir.wav'))
x1 = x[0:50000]
w = np.blackman(2001)
N = 2048
H = 500
t = -90
minSineDur = .01
maxnSines = 150
freqDevOffset = 20
freqDevSlope = 0.02
Ns = 512
H = Ns//4
tfreq, tmag, tphase = SM.sineModelAnal(x1, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

numFrames = int(tfreq[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)
maxplotfreq = 3000.0

plt.figure(1, figsize=(9, 7))

plt.subplot(3,1,1)
plt.plot(np.arange(x1.size)/float(fs), x1, 'b', lw=1.5)
plt.axis([0,x1.size/float(fs),min(x1),max(x1)])
plt.title('x (bendir.wav)')                        

plt.subplot(3,1,2)
tracks = tfreq*np.less(tfreq, maxplotfreq)
tracks[tracks<=0] = np.nan
plt.plot(frmTime, tracks, color='k', lw=1.5)
plt.autoscale(tight=True)
plt.title('f_t, sine frequencies')  

plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y, 'b', lw=1.5)
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('y')    

plt.tight_layout()
UF.wavwrite(y, fs, 'bendir-sine-synthesis.wav')
plt.savefig('sineModel-anal-synth.png')
plt.show()
