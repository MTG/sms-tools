import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import sineModel as SM
import dftModel as DFT
import stft as STFT
import utilFunctions as UF

# read the sound of the bendir
(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/bendir.wav'))

# compute and odd size hamming window with a length sufficient to be able to identify frequencies separated by 88Hz (4*fs/2001)
w = np.hamming(2001)

N = 2048             # fft size the next power of 2 bigger than the window size
H = 128              # hop size Ns/4
t = -80              # magnitude threshold quite low
minSineDur = .02     # only accept sinusoidal trajectores bigger than 20ms
maxnSines = 150      # track as many as 150 parallel sinusoids
freqDevOffset = 10   # frequency deviation allowed in the sinusoids from frame to frame at frequency 0
freqDevSlope = 0.001 # slope of the frequency deviation, higher frequencies have bigger deviation

# compute the magnitude and phase spectrogram of input sound
mX, pX = STFT.stftAnal(x, fs, w, N, H)

# compute the sinusoidal model
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

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

# synthesize the output sound from the sinusoidal representation
Ns = 512
y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

# write the output sound
UF.wavwrite(y, fs, 'bendir-sineModel.wav')

plt.tight_layout()
plt.show()
