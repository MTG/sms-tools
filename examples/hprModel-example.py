import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import harmonicModel as HM
import stft as STFT
import dftModel as DFT
import utilFunctions as UF
import sineModel as SM
  
# read the sax-phrase sound
(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/sax-phrase.wav'))
w = np.blackman(601)        # compute analysis window of odd size
N = 1024                    # fft size
t = -100                    # magnitude threshold used for peak detection
nH = 100                    # maximum number of harmonic to detect
minf0 = 350                 # minimum fundamental frequency
maxf0 = 700                 # maximum fundamental frequency
f0et = 5                    # maximum error allowed in f0 detection algorithm
minSineDur = .1             # min size of harmonic tracks
harmDevSlope = 0.01         # allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
Ns = 512                    # fft size used in synthesis
H = Ns/4                    # hop size used in analysis and synthesis
  
# find harmonics
hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
  
# subtract harmonics from original sound
xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
  
# compute spectrogram of residual sound
mXr, pXr = STFT.stftAnal(xr, fs, hamming(Ns), Ns, H)
  
# synthesize harmonic component
yh = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)

# write harmonic and residual components
UF.wavwrite(yh, fs, 'sax-phrase-harmonic.wav')
UF.wavwrite(xr, fs, 'sax-phrase-residual.wav')

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
