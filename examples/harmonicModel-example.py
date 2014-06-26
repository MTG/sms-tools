import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
import harmonicModel as HM
import dftModel as DFT
import stft as STFT
import utilFunctions as UF
import sineModel as SM

(fs, x) = UF.wavread('../../sounds/vignesh.wav')   # read vignesh sound file
w = np.blackman(1201)                              # create odd size window
N = 2048                                           # fft size
t = -90                                            # magnitude threshold used for peak detection
nH = 100                                           # maximum number of harmonics to identify
minf0 = 130                                        # minimum fundamental frequency in sound
maxf0 = 300                                        # maximum fundamental frequency in sound
f0et = 7                                           # maximum error accepted in f0 detection algorithm
Ns = 512                                           # fft size used for synthesis
H = Ns/4                                           # hop size used in analysis and synthesis, has to be 1/4 of Ns
minSineDur = .1                                    # minimum duration of sinusoidal tracks
harmDevSlope = 0.01                                # allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation

# compute spectrogram of input sound
mX, pX = STFT.stftAnal(x, fs, w, N, H)

# computer harmonics of input sound
hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

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

y = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)  # synthesize harmonics
UF.wavwrite(y, fs, 'vignesh-harmonicModel.wav')        # write output sound

plt.tight_layout()
plt.show()

