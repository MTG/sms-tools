import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import dftAnal
import smsWavplayer as wp
import smsPeakProcessing as PP

(fs, x) = wp.wavread('../../../sounds/oboe-A4.wav')
N = 512*2
M = 511
t = -60
w = np.hamming(M)
start = .8*fs
plt.figure(1)
hN = N/2
hM = (M+1)/2

x1 = x[start:start+M]
mX, pX = dftAnal.dftAnal(x1, w, N)
ploc = PP.peakDetection(mX, hN, t)
iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc) 
pmag = mX[ploc]
freqaxis = fs*np.arange(N/2)/float(N)

plt.figure(1)
plt.subplot (2,1,1)
plt.plot(freqaxis,mX,'r')
plt.axis([300,2500,-70,max(mX)])
plt.plot(fs * iploc / N, ipmag, marker='x', color='b', linestyle='') 
plt.title('Spectral peaks: magnitude (oboe-A4.wav), zero padding = 2')      

plt.subplot (2,1,2)
plt.plot(freqaxis,pX,'c')
plt.axis([300,2500,min(pX),-6])
plt.plot(fs * iploc / N, ipphase, marker='x', color='b', linestyle='')   
plt.title('Spectral peaks: phase') 
plt.show()

