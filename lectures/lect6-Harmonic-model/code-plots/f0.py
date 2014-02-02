import numpy as np
import copy
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.fftpack import fft, ifft


def peak_interp(mX, pX, ploc):
  # mX: magnitude spectrum, pX: phase spectrum, ploc: locations of peaks
  # iploc, ipmag, ipphase: interpolated values
  
  val = mX[ploc]                                          # magnitude of peak bin 
  lval = mX[ploc-1]                                       # magnitude of bin at left
  rval = mX[ploc+1]                                       # magnitude of bin at right
  iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
  ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
  ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks

  return iploc, ipmag, ipphase

def peak_detection(mX, hN, t):
  # mX: magnitude spectrum, hN: half number of samples, t: threshold
  # to be a peak it has to accomplish three conditions:

  thresh = np.where(mX[1:hN-1]>t, mX[1:hN-1], 0);
  next_minor = np.where(mX[1:hN-1]>mX[2:], mX[1:hN-1], 0)
  prev_minor = np.where(mX[1:hN-1]>mX[:hN-2], mX[1:hN-1], 0)
  ploc = thresh * next_minor * prev_minor
  ploc = ploc.nonzero()[0] + 1

  return ploc

import numpy as np
import copy

def f0DetectionTwm(ploc, pmag, N, fs, ef0max, minf0, maxf0):
  # Fundamental frequency detection function from a series of spectral peak values
  # ploc, pmag: peak loc and mag, N: size of complex spectrum, fs: sampling rate,
  # ef0max: maximum error allowed, minf0: minimum f0, maxf0: maximum f0
  # returns f0: fundamental frequency in Hz
  
  nPeaks = ploc.size                                  # number of peaks available
  f0 = 0                                              # initialize output
  maxnpeaks = min (10, nPeaks)                        # maximum number of peaks to use
  if maxnpeaks > 3 :                                  # only find fundamental if 3 peaks exist
    pfreq = ploc/N*fs                                 # frequency in Hertz of peaks
    zvalue = min(pfreq)
    zindex = np.argmin(pfreq)

    if zvalue == 0 :                                  # avoid zero frequency peak
      pfreq[zindex] = 1
      pmag[zindex] = -100

    pmag_temp = copy.deepcopy(pmag)
    Mmag = max(pmag_temp)
    Mloc1 = np.argmax(pmag_temp)                      # find peak with maximum magnitude
    pmag_temp[Mloc1] = -100                           # clear max peak
    Mloc2 = np.argmax(pmag_temp)                      # find second maximum magnitude peak
    pmag_temp[Mloc2] = -100                           # clear second max peak
    Mloc3 = np.argmax(pmag_temp)                      # find third maximum magnitude peak
    pmag_temp[Mloc3] = -100                           # clear second max peak
    nCand = 6                                         # number of possible f0 candidates for each max peak
    f0c = np.zeros(3*nCand)                           # initialize array of candidates
    
    f0c[:nCand] = (pfreq[Mloc1]*np.ones(nCand)) / (np.arange(nCand, 0, -1)) # f0 candidates 
    f0c[nCand:nCand*2] = (pfreq[Mloc2]*np.ones(nCand)) / (np.arange(nCand, 0, -1)) 
    f0c[nCand*2:nCand*3] = (pfreq[Mloc3]*np.ones(nCand)) / (np.arange(nCand, 0, -1))
    
    f0c = np.array([x for x in f0c if x<maxf0 and x>minf0]) # candidates within boundaries

    if not f0c.size :                                 # if no candidates exit
      f0 = 0 
      f0error = 100
      return f0
    else :
      f0error = 0  

    f0, f0error = TWM(pfreq, pmag, maxnpeaks, f0c)    # call the TWM function
    if f0>0 and f0error>ef0max :                      # limit the possible error by ethreshold
      f0 = 0

  return f0 

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


# example run
(fs, x) = read('oboe.wav')
N = 1024
M = 256
t = -40
start = .8*fs
xw = x[start:start+M] * np.hamming(M)
X = fft(xw,N)
mX = 20 * np.log10(abs(X[0:N/2])/N)  
pX = np.unwrap( np.angle(X[:N/2]) ) 
ploc = peak_detection(mX, N/2, t)
pmag = mX[ploc]
iploc, ipmag, ipphase = peak_interp(mX, pX, ploc)
minf0 = 80.0
maxf0 = 2000.0
ipfreq = iploc/N*fs 
f0cand = np.arange(minf0, maxf0, 1.0)
maxnpeaks = 10
f0, ErrorPM, ErrorMP, Error = TWM (ipfreq, ipmag, maxnpeaks, f0cand)

plt.figure(1)
freq = np.arange(0, fs/2, fs/N)                     # frequency axis in Hz
freq = freq[:freq.size-1] 
plt.subplot (2,1,1)
plt.plot(freq, mX)
plt.plot(ipfreq, ipmag, 'r*')
plt.axvline(f0, color='r')
plt.axis([minf0,5000.0,0,max(mX)+2])
plt.title('Magnitude spectrum with peaks')

plt.subplot (2,1,2)
freq = np.arange(minf0, maxf0, 1.0)                     # frequency axis in Hz
plt.plot(ErrorPM[0], 'b')
plt.plot(ErrorMP, 'g')
plt.plot(Error[0], color='black')
plt.axvline(f0-minf0, color='r')
plt.axis([minf0,maxf0,-1,50])
plt.title('Errors. black: total; blue: predited to measured; green: measured to predicted')

plt.show()

