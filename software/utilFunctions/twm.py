import numpy as np
import copy

def f0DetectionTwm(pfreq, pmag, N, fs, ef0max, minf0, maxf0, maxnpeaks=10):
  # Fundamental frequency detection function from a series of spectral peak values
  # ploc, pmag: peak freq and mag, N: size of complex spectrum, fs: sampling rate,
  # ef0max: maximum error allowed, minf0: minimum f0, maxf0: maximum f0
  # returns f0: fundamental frequency in Hz
  
  nPeaks = pfreq.size                                  # number of peaks available
  f0 = 0                                              # initialize output
  maxnpeaks = min (maxnpeaks, nPeaks)                 # maximum number of peaks to use
  if maxnpeaks > 3 :                                  # only find fundamental if 3 peaks exist
    zvalue = min(pfreq)
    zindex = np.argmin(pfreq)

    if zvalue == 0 :                                  # avoid zero frequency peak
      pfreq[zindex] = 0
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
  MaxNMP = min(maxnpeaks, pfreq.size)
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

  return f0, Error[f0index]

