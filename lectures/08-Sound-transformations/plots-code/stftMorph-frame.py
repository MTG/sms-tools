import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
from scipy.signal import hamming, resample
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import dftModel as DFT
import utilFunctions as UF
import math

(fs, x1) = UF.wavread('../../../sounds/orchestra.wav')
(fs, x2) = UF.wavread('../../../sounds/speech-male.wav')
w1 = np.hamming(1024)
N1 = 1024
H1 = 256
w2 = np.hamming(1024)
N2 = 1024
smoothf = .1
balancef = .7

M1 = w1.size                                     # size of analysis window
hM1_1 = int(math.floor((M1+1)/2))                # half analysis window size by rounding
hM1_2 = int(math.floor(M1/2))                    # half analysis window size by floor
M2 = w2.size                                     # size of analysis window
hM2_1 = int(math.floor((M2+1)/2))                # half analysis window size by rounding
hM2_2 = int(math.floor(M2/2))                    # half analysis window size by floor2
loc1 = 14843
loc2 = 9294

x1 = x1[loc1-hM1_1:loc1+hM1_2]
x2 = x2[loc2-hM2_1:loc2+hM2_2]
mX1, pX1 = DFT.dftAnal(x1, w1, N1)           # compute dft
mX2, pX2 = DFT.dftAnal(x2, w2, N2)           # compute dft
# morph
mX2smooth = resample(np.maximum(-200, mX2), int(mX2.size*smoothf))  # smooth spectrum of second sound
mX2 = resample(mX2smooth, mX2.size) 
mY = balancef * mX2 + (1-balancef) * mX1                            # generate output spectrum
#-----synthesis-----
y = DFT.dftSynth(mY, pX1, M1) * sum(w1)  # overlap-add to generate output sound
mY1, pY1 = DFT.dftAnal(y, w1, M1)  # overlap-add to generate output sound

plt.figure(1, figsize=(12, 9))
plt.subplot(321)
plt.plot(np.arange(N1)/float(fs), x1*w1, 'b', lw=1.5)
plt.axis([0, N1/float(fs), min(x1*w1), max(x1*w1)])
plt.title('x1 (orchestra.wav)')

plt.subplot(323)
plt.plot(fs*np.arange(mX1.size)/float(mX1.size), mX1-max(mX1), 'r', lw=1.5, label = 'mX1')
plt.plot(fs*np.arange(mX2.size)/float(mX2.size), mX2-max(mX2), 'k', lw=1.5, label='mX2')
plt.legend(prop={'size':10})
plt.axis([0,fs/4.0,-70,2])
plt.title('mX1 + mX2 (speech-male.wav)')

plt.subplot(325)
plt.plot(fs*np.arange(pX1.size)/float(pX1.size), pX1, 'c', lw=1.5)
plt.axis([0,fs/4.0,min(pX1),20])
plt.title('pX1')

plt.subplot(322)
plt.plot(np.arange(N1)/float(fs), y, 'b', lw=1.5)
plt.axis([0, float(N1)/fs, min(y), max(y)])
plt.title('y')

plt.subplot(324)
plt.plot(fs*np.arange(mY1.size)/float(mY1.size), mY1-max(mY1), 'r', lw=1.5)
plt.axis([0,fs/4.0,-70,2])
plt.title('mY')

plt.subplot(326)
plt.plot(fs*np.arange(pY1.size)/float(pY1.size), pY1, 'c', lw=1.5)
plt.axis([0,fs/4.0,min(pY1),6])
plt.title('pY')

plt.tight_layout()

plt.savefig('stftMorph-frame.png')
plt.show()
