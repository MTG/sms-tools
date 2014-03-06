import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time
from scipy.fftpack import fft, ifft
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import waveIO as WIO

    
lpc = ess.LPC(order=14)
N= 512
(fs, x) = WIO.wavread('../../../sounds/soprano-E4.wav')
first = 20000
last = first+N
x1 = x[first:last]
X = fft(hamming(N)*x1) 
mX = 20 * np.log10(abs(X[:N/2]))

coeff = lpc(x1)
Y = fft(coeff[0], N) 
mY = 20 * np.log10(abs(Y[:N/2]))

  
plt.figure(1)
plt.subplot(2,1,1)
plt.plot(np.arange(first, last)/float(fs), x[first:last], 'b')
plt.axis([first/float(fs), last/float(fs), min(x[first:last]), max(x[first:last])])
plt.title('x (soprano-E4.wav)')

plt.subplot(2,1,2)
plt.plot(np.arange(0, fs/2.0, fs/float(N)), mX-max(mX), 'r', label="mX")
plt.plot(np.arange(0, fs/2.0, fs/float(N)), -mY-max(-mY)-3, 'k', label="mY")
plt.legend()
plt.axis([0, fs/2, -60, 3])
plt.title('magnitude spectrum and LPC approximation')

plt.show()
