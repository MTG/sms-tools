# example of using the functions in software/models/stochasticModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hanning, resample
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import stochasticModel as STM
import utilFunctions as UF

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
(fs, x) = UF.wavread('../sounds/ocean.wav') 

# hop size
H = 256 

# decimation factor used for the stochastic approximation
stocf = .1  

# --------- computation -----------------  

# compute stochastic model                                          
mYst = STM.stochasticModelAnal(x, H, stocf)             

# synthesize sound from stochastic model
y = STM.stochasticModelSynth(mYst, H)                   

# --------- plotting --------------------

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

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write output sound
UF.wavwrite(y, fs, 'ocean-stochasticModel.wav')       



