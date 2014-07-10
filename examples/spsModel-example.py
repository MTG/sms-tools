# example of using the functions in software/models/spsModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import sineModel as SM
import stft as STFT
import utilFunctions as UF
import stochasticModel as STM

# input sound (monophonic with sampling rate of 44100)
inputFile = '../sounds/bendir.wav'

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
window = 'hamming' 

# analysis window size 
M = 2001

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

# output sound files (monophonic with sampling rate of 44100)
outputFileSines = 'bendir_spsModel_sines.wav'
outputFileStochastic = 'bendir_sprModel_stochastic.wav'
outputFile = 'bendir_spsModel.wav'

# --------- computation -----------------

# read input sound
(fs, x) = UF.wavread(inputFile)

# compute analysis window
w = get_window(window, M)

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

# sum sinusoids and stochastic
y = yst[:min(yst.size, ys.size)]+ys[:min(yst.size, ys.size)]

# write sounds files for sinusoidal, residual, and the sum
UF.wavwrite(ys, fs, outputFileSines)
UF.wavwrite(yst, fs, outputFileStochastic)
UF.wavwrite(y, fs, outputFile)

# --------- plotting --------------------

# plot stochastic component
plt.figure(1, figsize=(12, 9)) 

# frequency range to plot
maxplotfreq = 10000.0

# plot the input sound
plt.subplot(3,1,1)
plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('input sound: x')


plt.subplot(3,1,2)
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
plt.xlabel('time(s)')
plt.ylabel('Frequency(Hz)')
plt.autoscale(tight=True)
plt.title('sinusoidal + stochastic spectrogram')

# plot the output sound
plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y)
plt.axis([0, y.size/float(fs), min(y), max(y)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('output sound: y')

plt.tight_layout()
plt.show()


