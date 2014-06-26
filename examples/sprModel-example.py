import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import sineModel as SM
import stft as STFT
import utilFunctions as UF
  

# read bendir sound
(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/bendir.wav'))
w = np.hamming(2001)          # compute analysis window of odd size
N = 2048                      # fft size
H = 128                       # hop size of analysis window
t = -100                      # magnitude threshold used for peak detection
minSineDur = .02              # minimum length of sinusoidal tracks in seconds
maxnSines = 200               # maximum number of paralell sinusoids
freqDevOffset = 10            # allowed deviation in Hz at lowest frequency used in the frame to frame tracking
freqDevSlope = 0.001          # increase factor of the deviation as the frequency increases

# perform sinusoidal analysis
tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
	
# subtract sinusoids from original 
xr = UF.sineSubtraction(x, N, H, tfreq, tmag, tphase, fs)
  
# compute spectrogram of residual
mXr, pXr = STFT.stftAnal(xr, fs, hamming(H*2), H*2, H)
Ns = 512
	
# synthesize sinusoids
ys = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

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

# write sounds files for sinusoidal sound and residual sound
UF.wavwrite(ys, fs, 'bendir-sines.wav')
UF.wavwrite(xr, fs, 'bendir-residual.wav')

plt.tight_layout()
plt.show()
