import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import sys, os, functools, time
from scipy.fftpack import fft, ifft, fftshift
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import dftModel as DFT
import utilFunctions as UF

(fs, x) = UF.wavread('../../../sounds/oboe-A4.wav')
M = 601
w = np.blackman(M)
N = 1024
hN = N//2
Ns = 512
hNs = Ns//2
H = Ns//4
pin = 5000
t = -70
x1 = x[pin:pin+w.size]
mX, pX = DFT.dftAnal(x1, w, N)
ploc = UF.peakDetection(mX, t)
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)
freqs = iploc*fs/N 
Y = UF.genSpecSines(freqs, ipmag, ipphase, Ns, fs)       
mY = 20*np.log10(abs(Y[:hNs]))
pY = np.unwrap(np.angle(Y[:hNs]))
y= fftshift(ifft(Y))*sum(blackmanharris(Ns))
sw = np.zeros(Ns) 
ow = triang(2*H);    
sw[hNs-H:hNs+H] = ow  
bh = blackmanharris(Ns)     
bh = bh / sum(bh)     
sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]  


plt.figure(1, figsize=(9, 6))

plt.subplot(3,1,1)
plt.plot(np.arange(hNs), mY, 'r', lw=1.5)
plt.axis([0, hNs,-90,max(mY)+2])
plt.title("mY, Blackman-Harris, Ns = 512")

plt.subplot(3,1,2)
plt.plot(np.arange(-hNs,hNs), y, 'b', lw=1.5)
plt.plot(np.arange(-hNs,hNs), max(y)*bh/max(bh), 'k', alpha=.5,lw=1.5)
plt.axis([-hNs, hNs,min(y),max(y)+.1])
plt.title("y, size = Ns = 512 (Blackman-Harris window)")

yw = y * sw / max(sw)
plt.subplot(3,1,3)
plt.plot(np.arange(-hNs,hNs), yw, 'b',lw=1.5)
plt.plot(np.arange(-hNs/2,hNs/2), max(y)*ow/max(ow), 'k', alpha=.5,lw=1.5)
plt.axis([-hNs, hNs,min(yw),max(yw)+.1])
plt.title("yw = y * triangular / Blackman Harris; size = Ns/2 = 256")

plt.tight_layout()
plt.savefig('synthesis-window.png')
plt.show()
