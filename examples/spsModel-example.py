import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import utilFunctions as UF
import sineModel as SM
import stochasticModel as STM
	
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
stocf = 0.2                   # decimation factor used for the stochastic approximation
	
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

# plot stochastic component
plt.figure(1, figsize=(9.5, 7)) 
maxplotfreq = 22500.0
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
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.autoscale(tight=True)
plt.title('sinusoidal + residual components')

# write sounds files for sinusoidal sound and stochastic sound
UF.wavwrite(ys, fs, 'bendir-sines.wav')
UF.wavwrite(yst, fs, 'bendir-stochastic.wav')

plt.tight_layout()
plt.show()
