# example of using the functions in software/models/harmonicModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import sineModel as SM
import stft as STFT
import utilFunctions as UF
import harmonicModel as HM

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
inputFile = '../sounds/vignesh.wav'

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
window = 'blackman' 

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

# output sound file (monophonic with sampling rate of 44100)
outputFile = 'vignesh_harmonicModel.wav'  

# --------- computation -----------------

# read input sound
(fs, x) = UF.wavread(inputFile)

# compute analysis window
w = get_window(window, M)

# compute spectrogram of input sound
mX, pX = STFT.stftAnal(x, fs, w, N, H)

# computer harmonics of input sound
hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

# synthesize harmonics
y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)  

# write the sound resulting from the inverse stft
UF.wavwrite(y, fs, outputFile)

# --------- plotting --------------------

# create figure to show plots
plt.figure(1, figsize=(12, 9))

# frequency range to plot
maxplotfreq = 5000.0

# plot the input sound
plt.subplot(3,1,1)
plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('input sound: x')

# plot magnitude spectrogram
plt.subplot(3,1,2)
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
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('magnitude spectrogram + harmonic tracks')
plt.autoscale(tight=True)

# plot the output sound
plt.subplot(3,1,3)
plt.plot(np.arange(y.size)/float(fs), y)
plt.axis([0, y.size/float(fs), min(y), max(y)])
plt.ylabel('amplitude')
plt.xlabel('time (sec)')
plt.title('output sound: y')

plt.tight_layout()
plt.show()

      



