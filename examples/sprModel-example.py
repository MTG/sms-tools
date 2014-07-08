# example of using the functions in software/models/sprModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import sineModel as SM
import stft as STFT
import utilFunctions as UF

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

# perform sinusoidal analysis
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
	
# subtract sinusoids from original 
xr = UF.sineSubtraction(x, N, H, tfreq, tmag, tphase, fs)
  
# compute spectrogram of residual
mXr, pXr = STFT.stftAnal(xr, fs, hamming(H*2), H*2, H)
Ns = 512
	
# synthesize sinusoids
ys = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

# --------- plotting --------------------

# plot magnitude spectrogram of residual
plt.figure(1, figsize=(9.5, 7))
numFrames = int(mXr[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(H)*float(fs)/(H*2)                       
plt.pcolormesh(frmTime, binFreq, np.transpose(mXr))
plt.autoscale(tight=True)

# plot sinusoidal frequencies on top of residual spectrogram
tfreq[tfreq==0] = np.nan
numFrames = int(tfreq[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, tfreq, color='k', ms=3, alpha=1)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.autoscale(tight=True)
plt.title('sinusoidal + residual components')

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write sounds files for sinusoidal sound and residual sound
UF.wavwrite(ys, fs, 'bendir-sines.wav')
UF.wavwrite(xr, fs, 'bendir-residual.wav')


