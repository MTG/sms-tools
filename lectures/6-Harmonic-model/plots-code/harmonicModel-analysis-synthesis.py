import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import harmonicModelAnal as HMA
import harmonicModelSynth as HMS
import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH

try:
  import genSpecSines_C as GS
  import twm_C as fd
except ImportError:
  import genSpecSines as GS
  import twm as fd
  EH.printWarning(1)


(fs, x) = WIO.wavread('../../../sounds/vignesh.wav')
w = np.blackman(1201)
N = 2048
t = -90
nH = 100
minf0 = 130
maxf0 = 300
f0et = 7
maxnpeaksTwm = 4
Ns = 512
H = Ns/4

hfreq, hmag, hphase = HMA.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, maxnpeaksTwm)
y = HMS.harmonicModelSynth(hfreq, hmag, hphase, fs)

numFrames = int(hfreq[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)

plt.figure(1)

plt.subplot(3,1,1)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.axis([0,x.size/float(fs),min(x),max(x)])
plt.title('x = wavread("vignesh.wav")')                        

plt.subplot(3,1,2)
yhfreq = hfreq
yhfreq[hfreq==0] = np.nan
plt.plot(frmTime, hfreq, color='k')
plt.axis([0,y.size/float(fs),0,8000])
plt.title('harmonics')

plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y, 'b')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('y')    

plt.show()

