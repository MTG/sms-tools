# example of using the functions in software/models/hpsModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import harmonicModel as HM
import stft as STFT
import dftModel as DFT
import utilFunctions as UF
import sineModel as SM

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

# size of fft used in synthesis
Ns = 512

# hop size (has to be 1/4 of Ns)
H = 128


# --------- computation -----------------
  
# find harmonics
hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
  
# subtract harmonics from original sound
xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
  
# compute spectrogram of residual sound
mXr, pXr = STFT.stftAnal(xr, fs, hamming(Ns), Ns, H)
  
# synthesize harmonic component
yh = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)

# --------- plotting --------------------

# create figure to plot
plt.figure(1, figsize=(9.5, 7))

# plot residual spectrogram
maxplotfreq = 20000.0
numFrames = int(mXr[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(Ns*maxplotfreq/fs)/Ns                        
plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:,:Ns*maxplotfreq/fs+1]))
plt.autoscale(tight=True)

# plot harmonic frequencies on residual spectrogram
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.autoscale(tight=True)
plt.title('harmonic + residual components')

plt.tight_layout()
plt.show()

# --------- write output sound ---------

# write harmonic and residual components
UF.wavwrite(yh, fs, 'sax-phrase-harmonic.wav')
UF.wavwrite(xr, fs, 'sax-phrase-residual.wav')

