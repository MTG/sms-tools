import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import math
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import dftAnal as DFT
import waveIO as WIO
import peakProcessing as PP
import harmonicDetection as HD
import genSpecSines as GS
import twm as TWM

(fs, x) = WIO.wavread('../../../sounds/flute-A4.wav')
pos = .8*fs
M = 512
hM1 = int(math.floor((M+1)/2)) 
hM2 = int(math.floor(M/2)) 
first = pos - hM1
last = pos + hM2
x1 = x[first:last]
w = np.hamming(M)
N = 1024
t = -100
nH = 40
minf0 = 420
maxf0 = 460
f0et = 5
maxnpeaksTwm = 5
minSineDur = .1
harmDevSlope = 0.01
Ns = 512
H = Ns/4
x2 = x[pos-Ns/2:pos+Ns/2]

mX, pX = DFT.dftAnal(x1, w, N)
ploc = PP.peakDetection(mX, N/2, t)
iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc) 
ipfreq = fs*iploc/N
f0 = TWM.f0DetectionTwm(ipfreq, ipmag, N, fs, f0et, minf0, maxf0, maxnpeaksTwm)
hfreqp = []
hfreq, hmag, hphase = HD.harmonicDetection(ipfreq, ipmag, ipphase, f0, nH, hfreqp, fs, harmDevSlope)
Yh = GS.genSpecSines(Ns*hfreq/fs, hmag, hphase, Ns) 
mYh = 20 * np.log10(abs(Yh[:Ns/2]))     
pYh = np.unwrap(np.angle(Yh[:Ns/2])) 
bh=blackmanharris(Ns)
X = fft(fftshift(x2*bh/sum(bh)))        
Xr = X-Yh 
mXr = 20 * np.log10(abs(Xr[:Ns/2]))     
pXr = np.unwrap(np.angle(Xr[:Ns/2])) 
xrw = np.real(fftshift(ifft(Xr))) * H * 2
yhw = np.real(fftshift(ifft(Yh))) * H * 2

maxplotfreq = 8000.0
 


plt.figure(1)
plt.subplot(3,2,1)
plt.plot(np.arange(M), x[first:last]*w, lw=1.5)
plt.axis([0, M, min(x[first:last]*w), max(x[first:last]*w)])
plt.title('x (flute-A4.wav)')

plt.subplot(3,2,3)
binFreq = (fs/2.0)*np.arange(mX.size)/(mX.size) 
plt.plot(binFreq,mX,'r', lw=1.5)
plt.axis([0,maxplotfreq,-90,max(mX)+2])
plt.plot(hfreq, hmag, marker='x', color='b', linestyle='') 
plt.title('mX and harmonics') 

plt.subplot(3,2,5)
plt.plot(binFreq,pX,'c', lw=1.5)
plt.axis([0,maxplotfreq,min(pX),33])
plt.plot(hfreq, hphase, marker='x', color='b', linestyle='')   
plt.title('pX and harmonics') 

plt.subplot(3,2,4)
binFreq = (fs/2.0)*np.arange(mXr.size)/(mXr.size) 
plt.plot(binFreq,mYh,'r', lw=.8, label='mYh')
plt.plot(binFreq,mXr,'r', lw=1.5, label='mXr')
plt.axis([0,maxplotfreq,-90,max(mYh)+2])
plt.legend(prop={'size':10})
plt.title('mYh and mXr') 

plt.subplot(3,2,6)
binFreq = (fs/2.0)*np.arange(mXr.size)/(mXr.size) 
plt.plot(binFreq,pYh,'c', lw=.8, label='pYh')
plt.plot(binFreq,pXr,'c', lw=1.5, label ='pXr')
plt.axis([0,maxplotfreq,-18,10])
plt.legend(prop={'size':10})
plt.title('pYh and pXr') 

plt.subplot(3,2,2)
plt.plot(np.arange(Ns), yhw, 'b', lw=.8, label='yh')
plt.plot(np.arange(Ns), xrw, 'b', lw=1.5, label='xr')
plt.axis([0, Ns, min(yhw), max(yhw)])
plt.legend(prop={'size':10})
plt.title('yh and xr')

plt.show()

