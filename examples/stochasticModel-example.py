import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hanning, resample
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import stochasticModel as STM
import utilFunctions as UF
	
(fs, x) = UF.wavread('../sounds/ocean.wav')             # read ocean sound
H = 256                                                 # hop size of analysis window
stocf = .1                                              # decimation factor used for the stochastic approximation
mYst = STM.stochasticModelAnal(x, H, stocf)             # compute stochastic model

# plot stochastic representation
plt.figure(1, figsize=(9.5, 7)) 
numFrames = int(mYst[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(stocf*H)*float(fs)/(stocf*2*H)                      
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst))
plt.autoscale(tight=True)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.title('stochastic approximation')

y = STM.stochasticModelSynth(mYst, H)                   # synthesize sound from stochastic model
UF.wavwrite(y, fs, 'ocean-stochasticModel.wav')         # write output sound

plt.tight_layout()
plt.show()

