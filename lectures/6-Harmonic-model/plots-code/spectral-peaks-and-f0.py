import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import dftAnal
import waveIO as WIO
import peakProcessing as PP

(fs, x) = WIO.wavread('../../../sounds/oboe-A4.wav')
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
plt.plot(freqaxis,mX,'r')
plt.axis([0,7000,-80,max(mX)+1])
plt.plot(fs * iploc / N, ipmag, marker='x', color='b', linestyle='') 
harms = np.arange(1,20)*440.0
plt.vlines(harms, -80, max(mX)+1, color='g')
plt.title('Magnitude spectrum + peaks (oboe-A4.wav)')      

plt.show()

