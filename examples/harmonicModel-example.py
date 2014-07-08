# example of using the functions in software/models/harmonicModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import harmonicModel as HM
import dftModel as DFT
import stft as STFT
import utilFunctions as UF
import sineModel as SM

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
(fs, x) = UF.wavread('../sounds/vignesh.wav') 

# analysis window size 
M = 1201

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
w = np.blackman(M) 

# fft size (power of two, bigger or equal than M)
N = 2048             

# magnitude threshold of spectral peaks
t = -90  

# minimun duration of sinusoidal tracks
minSineDur = .1

# maximum number of harmonics
nH = 100  

# minimum fundamental frequency in sound
minf0 = 130 

# maximum fundamental frequency in sound
maxf0 = 300 

# maximum error accepted in f0 detection algorithm                                    
f0et = 7                                         

# allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
harmDevSlope = 0.01 

# size of fft used in synthesis
Ns = 512

# hop size (has to be 1/4 of Ns)
H = 128

# --------- computation -----------------

# compute spectrogram of input sound
mX, pX = STFT.stftAnal(x, fs, w, N, H)

# computer harmonics of input sound
hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

# synthesize harmonics
y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)  

# --------- plotting --------------------

# create figure to show plots
plt.figure(1, figsize=(9.5, 7))

# plot magnitude spectrogmra
maxplotfreq = 20000.0                 # show onnly frequencies below this value
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N  
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.autoscale(tight=True)
  
# plot harmonics on top of spectrogram of input sound
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(hfreq[:,0].size)
plt.plot(frmTime, harms, color='k')
plt.autoscale(tight=True)
plt.title('mX + harmonics')

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write output sound
UF.wavwrite(y, fs, 'vignesh-harmonicModel.wav')        



