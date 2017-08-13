import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackman
import math
import sys, os, functools, time
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import dftModel as DFT
import utilFunctions as UF


def TWM (pfreq, pmag, maxnpeaks, f0c):
  # Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
  # pfreq, pmag: peak frequencies in Hz and magnitudes, maxnpeaks: maximum number of peaks used
  # f0cand: frequencies of f0 candidates
  # returns f0: fundamental frequency detected
  
  p = 0.5                                          # weighting by frequency value
  q = 1.4                                          # weighting related to magnitude of peaks
  r = 0.5                                          # scaling related to magnitude of peaks
  rho = 0.33                                       # weighting of MP error
  Amax = max(pmag)                                 # maximum peak magnitude
  
  harmonic = np.matrix(f0c)
  ErrorPM = np.zeros(harmonic.size)                 # initialize PM errors
  MaxNPM = min(maxnpeaks, pfreq.size)
  for i in range(0, MaxNPM) :                       # predicted to measured mismatch error
    difmatrixPM = harmonic.T * np.ones(pfreq.size)
    difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
    FreqDistance = np.amin(difmatrixPM, axis=1)     # minimum along rows
    peakloc = np.argmin(difmatrixPM, axis=1)
    Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
    PeakMag = pmag[peakloc]
    MagFactor = 10**((PeakMag-Amax)/20)
    ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
    harmonic = harmonic+f0c

  ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
  MaxNMP = min(10, pfreq.size)
  for i in range(0, f0c.size) :                    # measured to predicted mismatch error
    nharm = np.round(pfreq[:MaxNMP]/f0c[i])
    nharm = (nharm>=1)*nharm + (nharm<1)
    FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
    Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
    PeakMag = pmag[:MaxNMP]
    MagFactor = 10**((PeakMag-Amax)/20)
    ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

  Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
  f0index = np.argmin(Error)                       # get the smallest error
  f0 = f0c[f0index]                                # f0 with the smallest error

  return f0, ErrorPM, ErrorMP, Error

(fs, x) = UF.wavread('../../../sounds/oboe-A4.wav')
N = 1024
hN = N//2
M = 801
t = -40
start = int(.8*fs)
minf0 = 100
maxf0 = 1500
w = blackman (M)
x1 = x[start:start+M]
mX, pX = DFT.dftAnal(x1, w, N)          
ploc = UF.peakDetection(mX, t)    
iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc) 
ipfreq = fs * iploc/N
f0cand = np.arange(minf0, maxf0, 1.0)
maxnpeaks = 10
f0, ErrorPM, ErrorMP, Error = TWM (ipfreq, ipmag, maxnpeaks, f0cand)
freqaxis = fs*np.arange(mX.size)/float(N)

plt.figure(1, figsize=(9, 7))
plt.subplot (2,1,1)
plt.plot(freqaxis,mX,'r', lw=1.5)
plt.axis([100,5100,-80,max(mX)+1])
plt.plot(fs * iploc / N, ipmag, marker='x', color='b', linestyle='', markeredgewidth=1.5) 
plt.title('mX + peaks (oboe-A4.wav)')   

plt.subplot (2,1,2)
plt.plot(f0cand,ErrorPM[0], 'b', label = 'ErrorPM', lw=1.2)
plt.plot(f0cand,ErrorMP, 'g', label = 'ErrorMP', lw=1.2)
plt.plot(f0cand,Error, color='black', label = 'Error Total', lw=1.5)
plt.axis([minf0,maxf0,min(Error),130])
plt.legend()
plt.title('TWM Errors')

plt.tight_layout()
plt.savefig('f0-TWM-errors-1.png')
plt.show()

