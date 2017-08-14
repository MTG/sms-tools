import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
import math
import sys, os, time
from scipy.fftpack import fft, ifft
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import utilFunctions as UF

    
lpc = ess.LPC(order=14)
N= 512
(fs, x) = UF.wavread('../../../sounds/soprano-E4.wav')
first = 20000
last = first+N
x1 = x[first:last]
X = fft(hamming(N)*x1) 
mX = 20 * np.log10(abs(X[:N//2]))

coeff = lpc(x1)
Y = fft(coeff[0], N) 
mY = 20 * np.log10(abs(Y[:N//2]))

  
plt.figure(1, figsize=(9, 5))
plt.subplot(2,1,1)
plt.plot(np.arange(first, last)/float(fs), x[first:last], 'b', lw=1.5)
plt.axis([first/float(fs), last/float(fs), min(x[first:last]), max(x[first:last])])
plt.title('x (soprano-E4.wav)')

plt.subplot(2,1,2)
plt.plot(np.arange(0, fs/2.0, fs/float(N)), mX-max(mX), 'r', lw=1.5, label="mX")
plt.plot(np.arange(0, fs/2.0, fs/float(N)), -mY-max(-mY)-3, 'k', lw=1.5, label="mY")
plt.legend()
plt.axis([0, fs/2, -60, 3])
plt.title('mX + mY (LPC approximation)')

plt.tight_layout()
plt.savefig('LPC.png')
plt.show()
