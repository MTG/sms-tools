# example of using the functions in software/models/hpsModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import hpsModel as HPS

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
inputFile = '../sounds/sax-phrase.wav'

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
window = 'blackman' 

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

# output sound files (monophonic with sampling rate of 44100)
outputFileSines = 'sax_phrase_hpsModel_sines.wav'
outputFileStochastic = 'sax_phrase_hprModel_stochastic.wav'
outputFile = 'sax_phrase_hpsModel.wav'

# --------- computation -----------------

# read input sound
(fs, x) = UF.wavread(inputFile)

# compute analysis window
w = get_window(window, M)

# compute the harmonic plus stochastic model of the whole sound
hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
	
# synthesize a sound from the harmonic plus stochastic representation
y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, mYst, Ns, H, fs)

# write sounds files for harmonics, stochastic, and the sum
UF.wavwrite(yh, fs, outputFileSines)
UF.wavwrite(yst, fs, outputFileStochastic)
UF.wavwrite(y, fs, outputFile)

# --------- plotting --------------------

# create figure to plot
plt.figure(1, figsize=(12, 9))

# frequency range to plot
maxplotfreq = 15000.0

# plot the input sound
plt.subplot(3,1,1)
plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('input sound: x')

# plot spectrogram stochastic compoment
plt.subplot(3,1,2)
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
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.autoscale(tight=True)
plt.title('harmonics + stochastic spectrogram')

# plot the output sound
plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y)
plt.axis([0, y.size/float(fs), min(y), max(y)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('output sound: y')

plt.tight_layout()
plt.show()

