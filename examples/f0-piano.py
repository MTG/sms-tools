import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackman, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))

import utilFunctions as UF
import stft as STFT
import harmonicModel as HM
import sineModel as SM

def sinewaveSynth(freq, mag, N, H, fs):
  # Synthesis of a time-varying sinusoid
  # freq,mag, phase: frequency, magnitude and phase of sinusoid,
  # N: synthesis FFT size, H: hop size, fs: sampling rate
  # returns y: output array sound
  hN = N/2                                                # half of FFT size for synthesis
  L = freq.size                                           # number of frames
  pout = 0                                                # initialize output sound pointer         
  ysize = H*(L+3)                                         # output sound size
  y = np.zeros(ysize)                                     # initialize output array
  sw = np.zeros(N)                                        # initialize synthesis window
  ow = triang(2*H);                                       # triangular window
  sw[hN-H:hN+H] = ow                                      # add triangular window
  bh = blackmanharris(N)                                  # blackmanharris window
  bh = bh / sum(bh)                                       # normalized blackmanharris window
  sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
  lastfreq = freq[0]                                      # initialize synthesis frequencies
  phase = 0                                               # initialize synthesis phases 
  for l in range(L):                                      # iterate over all frames
    phase += (np.pi*(lastfreq+freq[l])/fs)*H              # propagate phases
    Y = UF.genSpecSines(freq[l], mag[l], phase, N, fs)    # generate sines in the spectrum         
    lastfreq = freq[l]                                    # save frequency for phase propagation
    yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
    y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
    pout += H                                             # advance sound pointer
  y = np.delete(y, range(hN))                             # delete half of first window
  y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window 
  return y

(fs, x) = UF.wavread('../sounds/piano.wav')
w = np.blackman(1501)
N = 2048
t = -90
minf0 = 100
maxf0 = 300
f0et = 1
maxnpeaksTwm = 4
H = 128
x1 = x[1.5*fs:1.8*fs]

plt.figure(1, figsize=(9, 7))
mX, pX = STFT.stftAnal(x, fs, w, N, H)
f0 = HM.f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et)
yf0 = sinewaveSynth(f0, np.zeros(f0.size), 512, 128, fs)
f0[f0==0] = np.nan
maxplotfreq = 800.0
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.autoscale(tight=True)
  
plt.plot(frmTime, f0, linewidth=2, color='k')
plt.autoscale(tight=True)
plt.title('mX + f0 (piano.wav), TWM')

plt.tight_layout()
plt.savefig('f0TWM-piano.png')
plt.show()

