import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import numpy as np
import time, os, sys
import math
from scipy.fftpack import fft, ifft, fftshift

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import smsWavplayer as wp
import dftAnal as DF
import smsPeakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"


(fs, x) = wp.wavread('../../../sounds/oboe-A4.wav')
M = 601
w = np.blackman(M)
N = 1024
hN = N/2
Ns = 512
hNs = Ns/2
pin = 5000
t = -70
x1 = x[pin:pin+w.size]
mX, pX = DF.dftAnal(x1, w, N)
ploc = PP.peakDetection(mX, hN, t)
iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)
ploc = iploc*Ns/N 
Y = GS.genSpecSines(ploc, ipmag, ipphase, Ns)       
mY = 20*np.log10(abs(Y[:hNs]))
pY = np.unwrap(np.angle(Y[:hNs]))
y= fftshift(ifft(Y))*sum(blackmanharris(Ns))

plt.figure(1)

plt.subplot(4,1,1)
plt.plot(np.arange(-M/2,M/2), x1, 'b')
plt.axis([-M/2,M/2, min(x1), max(x1)])
plt.title("x = wavread(oboe-A4.wav); M = 601")

plt.subplot(4,1,2)
plt.plot(np.arange(hN), mX, 'r')
plt.plot(iploc, ipmag, marker='x', color='b', linestyle='') 
plt.axis([0, hN,-90,max(mX)+2])
plt.title("mX + spectral peaks; Blackman, N = 1024")

plt.subplot(4,1,3)
plt.plot(np.arange(hNs), mY, 'r')
plt.axis([0, hNs,-90,max(mY)+2])
plt.title("mY; Blackman-Harris; Ns = 512")

plt.subplot(4,1,4)
plt.plot(np.arange(Ns), y, 'b')
plt.axis([0, Ns,min(y),max(y)])
plt.title("y; Ns = 512")

plt.show()
