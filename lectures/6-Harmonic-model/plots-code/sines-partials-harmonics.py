import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import waveIO as WIO
import dftAnal as DF
import peakProcessing as PP

(fs, x) = WIO.wavread('../../../sounds/sine-440+490.wav')
w = np.hamming(3529)
N = 16084*2
hN = N/2
t = -20
pin = 4850
x1 = x[pin:pin+w.size]
mX1, pX1 = DF.dftAnal(x1, w, N)
ploc = PP.peakDetection(mX1, hN, t)
pmag = mX1[ploc] 
iploc, ipmag, ipphase = PP.peakInterp(mX1, pX1, ploc)

plt.figure(1)
plt.subplot(311)
plt.plot(fs*np.arange(0,N/2)/float(N), mX1-max(mX1), 'r')
plt.plot(fs * iploc / N, ipmag-max(mX1), marker='x', color='b', alpha=1, linestyle='') 
plt.axis([200, 1000, -80, 4])
plt.title('Sines: sine-440+490.wav')

(fs, x) = WIO.wavread('../../../sounds/vibraphone-C6.wav')
w = np.blackman(401)
N = 1024
hN = N/2
t = -80
pin = 200
x2 = x[pin:pin+w.size]
mX2, pX2 = DF.dftAnal(x2, w, N)
ploc = PP.peakDetection(mX2, hN, t)
pmag = mX2[ploc] 
iploc, ipmag, ipphase = PP.peakInterp(mX2, pX2, ploc)

plt.subplot(3,1,2)
plt.plot(fs*np.arange(0,N/2)/float(N), mX2-max(mX2), 'r')
plt.plot(fs * iploc / N, ipmag-max(mX2), marker='x', color='b', alpha=1, linestyle='') 
plt.axis([500,10000,-100,4])
plt.title('Partials: vibraphone-C6.wav')

(fs, x) = WIO.wavread('../../../sounds/oboe-A4.wav')
w = np.blackman(651)
N = 2048
hN = N/2
t = -80
pin = 10000
x3 = x[pin:pin+w.size]
mX3, pX3 = DF.dftAnal(x3, w, N)
ploc = PP.peakDetection(mX3, hN, t)
pmag = mX3[ploc] 
iploc, ipmag, ipphase = PP.peakInterp(mX3, pX3, ploc)

plt.subplot(3,1,3)
plt.plot(fs*np.arange(0,N/2)/float(N), mX3-max(mX3), 'r')
plt.plot(fs * iploc / N, ipmag-max(mX3), marker='x', color='b', alpha=1, linestyle='')
plt.axis([0,6000,-70,2])
plt.title('Harmonics: oboe-A4.wav')

plt.show()
