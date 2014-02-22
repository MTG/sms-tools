import numpy as np
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import sineModelAnal, sineModelSynth
import waveIO as WIO
import peakProcessing as PP
import errorHandler as EH

try:
  import genSpecSines_C as GS
except ImportError:
  import genSpecSines as GS
  EH.printWarning(1)


(fs, x) = WIO.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../sounds/bendir.wav'))
w = np.hamming(1001)
N = 2048
t = -60
Ns = 1024
H = Ns/4
x1=x[0:50000]
ploc, pmag, pphase = sineModelAnal.sineModelAnal(x1, fs, w, N, H, t)
yploc = Ns * ploc / N
y = sineModelSynth.sineModelSynth(ploc, pmag, pphase, N, Ns, H)
numFrames = int(ploc[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)

plt.figure(1)

plt.subplot(3,1,1)
plt.plot(np.arange(x1.size)/float(fs), x1, 'b')
plt.axis([0,x1.size/float(fs),min(x1),max(x1)])
plt.title('x = wavread("bendir.wav")')                        

plt.subplot(3,1,2)
yploc = ploc
yploc[ploc==0] = np.nan
plt.plot(frmTime, fs*yploc/Ns, 'x', color='k')
plt.axis([0,y.size/float(fs),0,4000])
plt.title('spectral peak locations')

plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y, 'b')
plt.axis([0,y.size/float(fs),min(y),max(y)])
plt.title('y')    

plt.show()
