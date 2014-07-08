# example of using the functions in software/models/hpsModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris, hanning, resample
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import hpsModel as HPS
import utilFunctions as UF

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
(fs, x) = UF.wavread('../sounds/sax-phrase.wav') 

# analysis window size 
M = 601

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
w = np.blackman(M) 

# fft size (power of two, bigger or equal than M)
N = 1024             

# magnitude threshold of spectral peaks
t = -100  

# minimun duration of sinusoidal tracks
minSineDur = .1

# maximum number of harmonics
nH = 100  

# minimum fundamental frequency in sound
minf0 = 350 

# maximum fundamental frequency in sound
maxf0 = 700 

# maximum error accepted in f0 detection algorithm                                    
f0et = 5                                        

# allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
harmDevSlope = 0.01 

# decimation factor used for the stochastic approximation
stocf = .1  

# size of fft used in synthesis
Ns = 512

# hop size (has to be 1/4 of Ns)
H = 128


# --------- computation -----------------
	
# compute the harmonic plus stochastic model of the whole sound
hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
	
# synthesize a sound from the harmonic plus stochastic representation
y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, mYst, Ns, H, fs)

# --------- plotting --------------------

# plot spectrogram stochastic compoment
plt.figure(1, figsize=(9.5, 7)) 
maxplotfreq = 22500.0
numFrames = int(mYst[:,0].size)
sizeEnv = int(mYst[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:sizeEnv*maxplotfreq/(.5*fs)+1]))
plt.autoscale(tight=True)

# plot harmonic on top of stochastic spectrogram
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.autoscale(tight=True)
plt.title('harmonic + stochastic components')

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write output sound and harmonic and stochastic components
UF.wavwrite(y, fs, 'sax-phrase-hpsModel.wav')
UF.wavwrite(yh, fs, 'sax-phrase-harmonics.wav')
UF.wavwrite(yst, fs, 'sax-phrase-stochastic.wav')
