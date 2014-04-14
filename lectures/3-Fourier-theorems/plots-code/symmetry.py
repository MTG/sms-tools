import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.fftpack import fft, ifft, fftshift
import math

sys.path.append('../../../software/models/')

import utilFunctions as UF
import dftModel as DF
(fs, x) = UF.wavread('../../../sounds/soprano-E4.wav')
w = np.hamming(511)
N = 512
pin = 5000
hM1 = int(math.floor((w.size+1)/2)) 
hM2 = int(math.floor(w.size/2)) 
fftbuffer = np.zeros(N)  
x1 = x[pin-hM1:pin+hM2]
xw = x1*w
fftbuffer[:hM1] = xw[hM2:]
fftbuffer[N-hM2:] = xw[:hM2]        
X = fftshift(fft(fftbuffer))
mX = 20 * np.log10(abs(X))       
pX = np.unwrap(np.angle(X))

plt.figure(1, figsize=(9.5, 7))
plt.subplot(311)
plt.plot(np.arange(-hM1, hM2), x1, lw=1.5)
plt.axis([-hM1, hM2, min(x1), max(x1)])
plt.ylabel('amplitude')
plt.title('x (soprano-E4.wav)')

plt.subplot(3,1,2)
plt.plot(np.arange(-N/2,N/2), mX, 'r', lw=1.5)
plt.axis([-N/2,N/2,-48,max(mX)])
plt.title ('mX = 20*log10(abs(X))')
plt.ylabel('amplitude (dB)')

plt.subplot(3,1,3)
plt.plot(np.arange(-N/2,N/2), pX, 'c', lw=1.5)
plt.axis([-N/2,N/2,min(pX),max(pX)])
plt.title ('pX = unwrap(angle(X))')
plt.ylabel('phase (radians)')

plt.tight_layout()
plt.savefig('symmetry.png')
plt.show()
