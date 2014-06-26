import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris, hanning, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import hpsModel as HPS
import utilFunctions as UF

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
stocf = .2                  # decimation factor used for the stochastic approximation
	
# compute the harmonic plus stochastic model of the whole sound
hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
	
# synthesize a sound from the harmonic plus stochastic representation
y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, hphase, mYst, Ns, H, fs)
 
# write output sound and harmonic and stochastic components
UF.wavwrite(y, fs, 'sax-phrase-hpsModel.wav')
UF.wavwrite(yh, fs, 'sax-phrase-harmonics.wav')
UF.wavwrite(yst, fs, 'sax-phrase-stochastic.wav')

# plot spectrogram stochastic compoment
plt.figure(1, figsize=(9.5, 7)) 
maxplotfreq = 20000.0
numFrames = int(mYst[:,0].size)
sizeEnv = int(mYst[0,:].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = (.5*fs)*np.arange(sizeEnv*maxplotfreq/(.5*fs))/sizeEnv                      
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:sizeEnv*maxplotfreq/(.5*fs)+1]))
plt.autoscale(tight=True)

# plot harmonic on top of stochastic spectrogram
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
plt.xlabel('Time(s)')
plt.ylabel('Frequency(Hz)')
plt.autoscale(tight=True)
plt.title('harmonic + stochastic components')

plt.tight_layout()
plt.show()
