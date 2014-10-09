import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
import sys, os
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
import stft as STFT

N = 512
spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=N, type='hann')
mfcc = MFCC ()
(fs, x) = UF.wavread('../../../sounds/vignesh.wav')

mfccs = []
pin = 0
pend = x.size-N
H = 256

while pin<pend:             
  mX = spectrum(window(x[pin:pin+N]))
  mfcc_bands, mfcc_coeffs = mfcc(mX)
  mfccs.append(mfcc_coeffs)
  pin += H                     

plt.figure(1, figsize=(9, 7))

numFrames = int(mfccs[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = fs*np.arange(N)/float(N)                              
plt.pcolormesh(frmTime, binFreq, np.transpose(mfccs[:,:N*maxplotfreq/fs+1]))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')


plt.tight_layout()
plt.savefig('mfcc.png')
plt.show()

