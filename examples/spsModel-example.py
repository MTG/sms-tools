# example of using the functions in software/models/sprModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import sineModel as SM
import stochasticModel as STM

# input sound (monophonic with sampling rate of 44100)
(fs, x) = UF.wavread('../sounds/bendir.wav')

# analysis window size 
M = 2001

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
w = np.hamming(M) 

# fft size (power of two, bigger or equal than M)
N = 2048             

# magnitude threshold of spectral peaks
t = -80  

# minimun duration of sinusoidal tracks
minSineDur = .02

# maximum number of parallel sinusoids
maxnSines = 150  

# frequency deviation allowed in the sinusoids from frame to frame at frequency 0   
freqDevOffset = 10 

# slope of the frequency deviation, higher frequencies have bigger deviation
freqDevSlope = 0.001 

# decimation factor used for the stochastic approximation
stocf = .2  

# size of fft used in synthesis
Ns = 512

# hop size (has to be 1/4 of Ns)
H = 128

# --------- computation -----------------

# perform sinusoidal analysis
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
	
# subtract sinusoids from original sound
Ns = 512
xr = UF.sineSubtraction(x, Ns, H, tfreq, tmag, tphase, fs)
	
# compute stochastic model of residual
mYst = STM.stochasticModelAnal(xr, H, stocf)
	
# synthesize sinusoids
ys = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)
	
# synthesize stochastic component
yst = STM.stochasticModelSynth(mYst, H)

# --------- plotting --------------------

# plot stochastic component
plt.figure(1, figsize=(9.5, 7)) 
maxplotfreq = 22500.0
numFrames = int(mYst[:,0].size)
sizeEnv = int(mYst[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:sizeEnv*maxplotfreq/(.5*fs)+1]))
plt.autoscale(tight=True)

# plot sinusoidal frequencies on top of stochastic component
sines = tfreq*np.less(tfreq,maxplotfreq)
sines[sines==0] = np.nan
numFrames = int(sines[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, sines, color='k', ms=3, alpha=1)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.autoscale(tight=True)
plt.title('sinusoidal + residual components')

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write sounds files for sinusoidal sound and stochastic sound
UF.wavwrite(ys, fs, 'bendir-sines.wav')
UF.wavwrite(yst, fs, 'bendir-stochastic.wav')


