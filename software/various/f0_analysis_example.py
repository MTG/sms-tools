# example of computing the fundamental frequency of a sound

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackman, blackmanharris
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import stft as STFT
import harmonicModel as HM
import sineModel as SM

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
(fs, x) = UF.wavread('../../sounds/piano.wav')

# analysis window size 
M = 1501

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
w = np.blackman(M)

# fft size (power of two, bigger or equal than M)
N = 2048             

# magnitude threshold of spectral peaks
t = -90 

# minimum fundamental frequency in sound
minf0 = 100 

# maximum fundamental frequency in sound
maxf0 = 300 

# maximum error accepted in f0 detection algorithm                                    
f0et = 1                           

# hop size
H = 128

# --------- computation -----------------

# compute spectrogram of input sound
mX, pX = STFT.stftAnal(x, fs, w, N, H)

# compute the fundamental frequency
f0 = HM.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et)

# delete short f0 tracks
f0 = UF.cleaningTrack(f0, 5)

# synthesize the f0 as a sinewave
yf0 = SM.sinewaveSynth(f0, np.array([0.4]), H, fs)

# --------- plotting --------------------

plt.figure(1, figsize=(9, 7))
f0[f0==0] = np.nan
maxplotfreq = 800.0
numFrames = int(f0.size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.autoscale(tight=True)
plt.plot(frmTime, f0, linewidth=2, color='k')
plt.autoscale(tight=True)
plt.title('mX + f0 (piano.wav), TWM')
plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write the sinusoid of the fundamental
UF.wavwrite(yf0, fs, 'piano-f0.wav')
