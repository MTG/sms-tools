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
H = Ns/4
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
sw = np.zeros(Ns) 
ow = triang(2*H);    
sw[hNs-H:hNs+H] = ow  
bh = blackmanharris(Ns)     
bh = bh / sum(bh)     
sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]  


plt.figure(1)

plt.subplot(3,1,1)
plt.plot(np.arange(-hNs,hNs), y, 'b')
plt.plot(np.arange(-hNs,hNs), max(y)*bh/max(bh), 'k', alpha=.5)
plt.axis([-hNs, hNs,min(y),max(y)+.1])
plt.title("y; size = Ns = 512 (Blackman-Harris)")

plt.subplot(3,3,4)
plt.plot(np.arange(-hNs,hNs), bh/max(bh), 'k', alpha=.9)
plt.axis([-hNs, hNs,0,1])
plt.title("Blackman-Harris")

plt.subplot(3,3,5)
plt.plot(np.arange(-hNs/2,hNs/2), ow/max(ow), 'k', alpha=.9)
plt.axis([-hNs/2, hNs/2,0,1])
plt.title("triangular")

plt.subplot(3,3,6)
plt.plot(np.arange(-hNs/2,hNs/2), sw[hNs-H:hNs+H]/max(sw), 'k', alpha=.9)
plt.axis([-hNs/2, hNs/2,0,1])
plt.title("triangular / Blackman-Harris")

yw = y * sw / max(sw)
plt.subplot(3,1,3)
plt.plot(np.arange(-hNs,hNs), yw, 'b')
plt.plot(np.arange(-hNs/2,hNs/2), max(y)*ow/max(ow), 'k', alpha=.5)
plt.axis([-hNs/2, hNs/2,min(yw),max(yw)+.1])
plt.title("yw = y * triangular / Blackman Harris; size = Ns/2 = 256")


plt.show()
