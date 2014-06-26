import matplotlib.pyplot as plt
import numpy as np
import time, os, sys
from scipy.fftpack import fft, ifft
import math

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import dftModel as DFT

	
# read the sound of the oboe note
(fs, x) = UF.wavread('../sounds/oboe-A4.wav')
	
w = np.blackman(511) # use and odd size blackman window 
N = 1024             # fft size of a power of two length
pin = 5000           # sample of input sound to start reading
	
# find the two sides of the windo
hM1 = int(math.floor((w.size+1)/2)) 
hM2 = int(math.floor(w.size/2)) 
	
# get a fragment of the input sound 
x1 = x[pin-hM1:pin+hM2]
 
# compute the dft of the sound fragment
mX, pX = DFT.dftAnal(x1, w, N)

# compute the inverse dft of the spectrum
y = DFT.dftSynth(mX, pX, w.size)*sum(w)

# create figure
plt.figure(1, figsize=(9.5, 7))

# plot the sound fragment
plt.subplot(4,1,1)
plt.plot(np.arange(-hM1, hM2), x1)
plt.axis([-hM1, hM2, min(x1), max(x1)])
plt.ylabel('amplitude')
plt.title('input signal: x')

# plot the magnitude spectrum
plt.subplot(4,1,2)
plt.plot(np.arange(N/2), mX, 'r')
plt.axis([0,N/2,min(mX),max(mX)])
plt.title ('magnitude spectrum: mX')
plt.ylabel('amplitude (dB)')
plt.ylabel('frequency samples')

# plot the phase spectrum
plt.subplot(4,1,3)
plt.plot(np.arange(N/2), pX, 'c')
plt.axis([0,N/2,min(pX),max(pX)])
plt.title ('phase spectrum: pX')
plt.ylabel('phase (radians)')
plt.ylabel('frequency samples')

# plot the sound resulting from the inverse dft
plt.subplot(4,1,4)
plt.plot(np.arange(-hM1, hM2), y)
plt.axis([-hM1, hM2, min(y), max(y)])
plt.ylabel('amplitude')
plt.title('output signal: y')

plt.tight_layout()
plt.show()
