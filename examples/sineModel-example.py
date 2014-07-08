# example of using the functions in software/models/sineModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hann, hamming, triang, blackman, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import sineModel as SM
import dftModel as DFT
import stft as STFT
import utilFunctions as UF

# ------- analysis parameters -------------------

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

# size of fft used in synthesis
Ns = 512

# hop size (has to be 1/4 of Ns)
H = 128

# --------- computation -----------------

# compute the magnitude and phase spectrogram of input sound
mX, pX = STFT.stftAnal(x, fs, w, N, H)

# compute the sinusoidal model
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

# synthesize the output sound from the sinusoidal representation
y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

# --------- plotting --------------------

# create figure to show plots
plt.figure(1, figsize=(9.5, 7))
	
# plot the magnitude spectrogram
maxplotfreq = 5000.0
maxplotbin = int(N*maxplotfreq/fs)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
plt.autoscale(tight=True)
	
# plot the sinusoidal frequencies on top of the spectrogram
tracks = tfreq*np.less(tfreq, maxplotfreq)
tracks[tracks<=0] = np.nan
plt.plot(frmTime, tracks, color='k')
plt.autoscale(tight=True)
plt.title('mX + sinusoidal tracks')

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write the output sound
UF.wavwrite(y, fs, 'bendir-sineModel.wav')


