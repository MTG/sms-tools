import numpy as np
import copy

def f0detectiontwm(ploc, pmag, N, fs, ef0max, minf0, maxf0):
  # Fundamental frequency detection function
  # ploc, pmag: peak loc and mag, N: size of complex spectrum, fs: sampling rate,
  # ef0max: maximum error allowed, minf0: minimum f0, maxf0: maximum f0
  # f0: fundamental frequency detected in Hz
  
  nPeaks = ploc.size                   # number of peaks
  f0 = 0                               # initialize output

  if nPeaks > 3 :                      # at least 3 peaks in spectrum for trying to find f0
    nf0peaks = min(50,nPeaks)          # use a maximum of 50 peaks
    f0, f0error = TWM(ploc[:nf0peaks], pmag[:nf0peaks], N, fs, minf0, maxf0)
    if f0>0 and f0error>ef0max :       # limit the possible error by ethreshold
      f0 = 0

  return f0 

def TWM (ploc, pmag, N, fs, minf0, maxf0):
  # Two-way mismatch algorithm (by Beauchamp&Maher)
  # ploc, pmag: peak locations and magnitudes, N: size of complex spectrum
  # fs: sampling rate of sound, minf0: minimum f0, maxf0: maximum f0
  # f0: fundamental frequency detected, f0error: error measure 
  
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
  
  f0c[:nCand] = (pfreq[Mloc1]*np.ones(nCand)) / (np.arange(nCand, 0, -1)) # candidates 
  f0c[nCand:nCand*2] = (pfreq[Mloc2]*np.ones(nCand)) / (np.arange(nCand, 0, -1)) 
  f0c[nCand*2:nCand*3] = (pfreq[Mloc3]*np.ones(nCand)) / (np.arange(nCand, 0, -1))
  
  f0c = np.array([x for x in f0c if x<maxf0 and x>minf0]) # candidates within boundaries

  if not f0c.size :                                 # if no candidates exit
    f0 = 0 
    f0error = 100
    return f0, f0error
  else :
    f0error = 0  
  
  harmonic = np.matrix(f0c)
  ErrorPM = np.zeros(harmonic.size)                 # initialize PM errors
  MaxNPM = min(10, ploc.size)
  for i in range(0, MaxNPM) :                       # predicted to measured mismatch error
    difmatrixPM = harmonic.T * np.ones(pfreq.size)
    difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
    FreqDistance = np.amin(difmatrixPM, axis=1)     # minimum along rows
    peakloc = np.argmin(difmatrixPM, axis=1)
    Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-0.5))
    PeakMag = pmag[peakloc]
    MagFactor = 10**((PeakMag-Mmag)/20)
    ErrorPM = ErrorPM + (Ponddif + MagFactor*(1.4*Ponddif-0.5)).T
    harmonic = harmonic+f0c

  ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
  MaxNMP = min(10, pfreq.size)
  for i in range(0, f0c.size) :                    # measured to predicted mismatch error
    nharm = np.round(pfreq[:MaxNMP]/f0c[i])
    nharm = (nharm>=1)*nharm + (nharm<1)
    FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
    Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-0.5))
    PeakMag = pmag[:MaxNMP]
    MagFactor = 10**((PeakMag-Mmag)/20)
    ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(1.4*Ponddif-0.5)))

  Error = (ErrorPM/MaxNPM) + (0.3*ErrorMP/MaxNMP)  # total errors
  f0index = np.argmin(Error)                       # get the smallest error
  f0 = f0c[f0index]                                # f0 with the smallest error

  return f0, f0error