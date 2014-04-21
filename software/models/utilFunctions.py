import numpy as np
import copy
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, time
import wave
import pyaudio
import os, copy, sys
from scipy.io.wavfile import write
from scipy.io.wavfile import read

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), './utilFunctions_C/'))

try:
  import utilFunctions_C as UF_C
except ImportError:
  print "module could not be imported"


def printError(errorID):
    if errorID == 1:
        print "Error opening file"
        
        
def printWarning(warningID):
    if warningID ==1:
        print "\n"
        print "-------------------------------------------------------------------------------"
        print "Warning:"
        print "Cython modules for some of the core functions were not imported."
        print "The processing might be significantly slower in such case"
        print "Please refer to the README file for instructions to compile cython modules"
        print "https://github.com/MTG/sms-tools/blob/master/README.md"
        print "-------------------------------------------------------------------------------"
        print "\n"


def f0DetectionTwm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
  # function that wraps the f0 detection function TWM, selecting the possible f0 candidates
  # and calling the function TWM with them
  # pfreq, pmag: peak frequencies and magnitudes, 
  # ef0max: maximum error allowed, minf0, maxf0: minimum  and maximum f0
  # f0t: f0 of previous frame if stable
  # returns f0: fundamental frequency in Hz

  if (pfreq.size < 3) & (f0t == 0):             # quit if less than 3 peaks and not previous f0
    return 0
  
  f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0] # use only peaks within range
  if (f0c.size == 0):                           # quit if no peaks within range
    return 0
  f0cf = pfreq[f0c]                             # frequencies of candidates
  f0cm = pmag[f0c]                              # magnitude of candidates

  if f0t>0:                                     # if stable f0 in previous frame 
    shortlist = np.argwhere(np.abs(f0cf-f0t)<f0t/2.0)[:,0]   # use only peaks close to it
    maxc = np.argmax(f0cm)
    maxcfd = f0cf[maxc]%f0t
    if maxcfd > f0t/2:
      maxcfd = f0t - maxcfd
    if (maxc not in shortlist) and (maxcfd>(f0t/4)):# or the maximum magnitude peak is not a harmonic
      shortlist = np.append(maxc, shortlist)
    f0cf = f0cf[shortlist]                      # frequencies of candidates                     

  if (f0cf.size == 0):                          # quit if no candidates
    return 0

  f0, f0error = UF_C.twm(pfreq, pmag, f0cf)      # call the TWM function
  
  if (f0>0) and (f0error<ef0max):               # accept f0 if below max error allowed
    return f0
  else:
    return 0


def TWM_p(pfreq, pmag, f0c):
  # Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
  # pfreq, pmag: peak frequencies in Hz and magnitudes, 
  # f0c: frequencies of f0 candidates
  # returns f0: fundamental frequency detected
  p = 0.5                                          # weighting by frequency value
  q = 1.4                                          # weighting related to magnitude of peaks
  r = 0.5                                          # scaling related to magnitude of peaks
  rho = 0.33                                       # weighting of MP error
  Amax = max(pmag)                                 # maximum peak magnitude
  maxnpeaks = 10                                   # maximum number of peaks used
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

def harmonicDetection(pfreq, pmag, pphase, f0, nH, hfreqp, fs, harmDevSlope=0.01):
  # detection of the harmonics from a set of spectral peaks, finds the peaks that are closer
  # to the ideal harmonic series built on top of a fundamental frequency
  # pfreq, pmag, pphase: peak frequencies, magnitudes and phases
  # f0: fundamental frequency, nH: number of harmonics,
  # hfreqp: harmonic frequencies of previous frame,
  # fs: sampling rate, harmDevSlope: slope of change of the deviation allowed to perfect harmonic
  # returns hfreq, hmag, hphase: harmonic frequencies, magnitudes, phases
  if (f0<=0):
    return np.zeros(nH), np.zeros(nH), np.zeros(nH)
  hfreq = np.zeros(nH)                                 # initialize harmonic frequencies
  hmag = np.zeros(nH)-100                              # initialize harmonic magnitudes
  hphase = np.zeros(nH)                                # initialize harmonic phases
  hf = f0*np.arange(1, nH+1)                           # initialize harmonic frequencies
  hi = 0                                               # initialize harmonic index
  if hfreqp == []:
    hfreqp = hf
  while (f0>0) and (hi<nH) and (hf[hi]<fs/2):          # find harmonic peaks
    pei = np.argmin(abs(pfreq - hf[hi]))             # closest peak
    dev1 = abs(pfreq[pei] - hf[hi])                  # deviation from perfect harmonic
    dev2 = (abs(pfreq[pei] - hfreqp[hi]) if hfreqp[hi]>0 else fs) # deviation from previous frame
    threshold = f0/3 + harmDevSlope * pfreq[pei]
    if ((dev1<threshold) or (dev2<threshold)):       # accept peak if deviation is small
      hfreq[hi] = pfreq[pei]                       # harmonic frequencies
      hmag[hi] = pmag[pei]                         # harmonic magnitudes
      hphase[hi] = pphase[pei]                     # harmonic phases
    hi += 1                                          # increase harmonic index
  return hfreq, hmag, hphase


def stochasticResidual(x, N, H, sfreq, smag, sphase, fs, stocf):
  # subtract sinusoids from a sound
  # x: input sound, N: fft-size, H: hop-size
  # sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
  # returns mYst: stochastic approximation of residual 
  hN = N/2  
  x = np.append(np.zeros(hN),x)                    # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hN))                    # add zeros at the end to analyze last sample
  bh = blackmanharris(N)                           # synthesis window
  w = bh/ sum(bh)                                  # normalize synthesis window
  L = sfreq[:,0].size                              # number of frames   
  pin = 0
  for l in range(L):
    xw = x[pin:pin+N]*w                            # window the input sound                               
    X = fft(fftshift(xw))                          # compute FFT 
    Yh = UF_C.genSpecSines(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N)   # generate spec sines          
    Xr = X-Yh                                      # subtract sines from original spectrum
    mXr = 20*np.log10(abs(Xr[:hN]))                # magnitude spectrum of residual
    mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)  # decimate the mag spectrum                        
    if l == 0: 
      mYst = np.array([mXrenv])
    else:
      mYst = np.vstack((mYst, np.array([mXrenv])))
    pin += H                                       # advance sound pointer
  return mYst


def sineSubtraction(x, N, H, sfreq, smag, sphase, fs):
  # subtract sinusoids from a sound
  # x: input sound, N: fft-size, H: hop-size
  # sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
  # returns xr: residual sound 
  hN = N/2  
  x = np.append(np.zeros(hN),x)                    # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hN))                    # add zeros at the end to analyze last sample
  bh = blackmanharris(N)                           # synthesis window
  w = bh/ sum(bh)                                  # normalize synthesis window
  sw = np.zeros(N)    
  sw[hN-H:hN+H] = triang(2*H) / w[hN-H:hN+H]
  L = sfreq[:,0].size                              # number of frames   
  xr = np.zeros(x.size)                            # initialize output array
  pin = 0
  for l in range(L):
    xw = x[pin:pin+N]*w                            # window the input sound                               
    X = fft(fftshift(xw))                          # compute FFT 
    Yh = UF_C.genSpecSines(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N)   # generate spec sines          
    Xr = X-Yh                                      # subtract sines from original spectrum
    xrw = np.real(fftshift(ifft(Xr)))              # inverse FFT
    xr[pin:pin+N] += xrw*sw                        # overlap-add
    pin += H                                       # advance sound pointer
  xr = np.delete(xr, range(hN))                    # delete half of first window which was added in stftAnal
  xr = np.delete(xr, range(xr.size-hN, xr.size))   # delete half of last window which was added in stftAnal
  return xr

def cleaningSineTracks(tfreq, minTrackLength=3):
  # delete short fragments of a collections of sinusoidal tracks 
  # tfreq: frequency of tracks
  # minTrackLength: minimum duration of tracks in number of frames
  # returns tfreqn: frequency of tracks
  nFrames = tfreq[:,0].size         # number of frames
  nTracks = tfreq[0,:].size         # number of tracks in a frame
  for t in range(nTracks):          # iterate over all tracks
    trackFreqs = tfreq[:,t]         # frequencies of one track
    trackBegs = np.nonzero((trackFreqs[:nFrames-1] <= 0) # begining of track contours
                & (trackFreqs[1:]>0))[0] + 1
    if trackFreqs[0]>0:
      trackBegs = np.insert(trackBegs, 0, 0)
    trackEnds = np.nonzero((trackFreqs[:nFrames-1] > 0)  # end of track contours
                & (trackFreqs[1:] <=0))[0] + 1
    if trackFreqs[nFrames-1]>0:
      trackEnds = np.append(trackEnds, nFrames-1)
    trackLengths = 1 + trackEnds - trackBegs             # lengths of trach contours
    for i,j in zip(trackBegs, trackLengths):             # delete short track contours
      if j <= minTrackLength:
        trackFreqs[i:i+j] = 0
  return tfreq

def cleaningTrack(track, minTrackLength=3):
  # delete short fragments of one single track
  # track: array of values
  # minTrackLength: minimum duration of tracks in number of frames
  # returns cleanTrack: array of clean values
  nFrames = track.size         # number of frames
  cleanTrack = np.copy(track)
  trackBegs = np.nonzero((track[:nFrames-1] <= 0) # begining of track contours
                & (track[1:]>0))[0] + 1
  if track[0]>0:
    trackBegs = np.insert(trackBegs, 0, 0)
  trackEnds = np.nonzero((track[:nFrames-1] > 0)  & (track[1:] <=0))[0] + 1
  if track[nFrames-1]>0:
    trackEnds = np.append(trackEnds, nFrames-1)
  trackLengths = 1 + trackEnds - trackBegs           # lengths of trach contours
  for i,j in zip(trackBegs, trackLengths):           # delete short track contours
    if j <= minTrackLength:
      cleanTrack[i:i+j] = 0
  return cleanTrack

def sineTracking(pfreq, pmag, pphase, tfreq, freqDevOffset=20, freqDevSlope=0.01):
  # tracking sinusoids from one frame to the next
  # pfreq, pmag, pphase: frequencies and magnitude of current frame
  # tfreq: frequencies of incoming tracks
  # freqDevOffset: minimum frequency deviation at 0Hz 
  # freqDevSlope: slope increase of minimum frequency deviation
  # returns tfreqn, tmagn, tphasen: frequencies, magnitude and phase of tracks
  tfreqn = np.zeros(tfreq.size)                              # initialize array for output frequencies
  tmagn = np.zeros(tfreq.size)                               # initialize array for output magnitudes
  tphasen = np.zeros(tfreq.size)                             # initialize array for output phases
  pindexes = np.array(np.nonzero(pfreq), dtype=np.int)[0]    # indexes of current peaks
  incomingTracks = np.array(np.nonzero(tfreq), dtype=np.int)[0] # indexes of incoming tracks
  newTracks = np.zeros(tfreq.size, dtype=np.int) -1           # initialize to -1 new tracks
  magOrder = np.argsort(-pmag[pindexes])                      # order current peaks by magnitude
  pfreqt = np.copy(pfreq)                                     # copy current peaks to temporary array
  pmagt = np.copy(pmag)                                       # copy current peaks to temporary array
  pphaset = np.copy(pphase)                                   # copy current peaks to temporary array

  # continue incoming tracks
  if incomingTracks.size > 0:                                 # if incoming tracks exist
    for i in magOrder:                                        # iterate over current peaks
      if incomingTracks.size == 0:                            # break when no more incoming tracks
        break
      track = np.argmin(abs(pfreqt[i] - tfreq[incomingTracks]))   # closest incoming track to peak
      freqDistance = abs(pfreq[i] - tfreq[incomingTracks[track]]) # measure freq distance
      if freqDistance < (freqDevOffset + freqDevSlope * pfreq[i]):  # choose track if distance is small 
          newTracks[incomingTracks[track]] = i                      # assign peak index to track index
          incomingTracks = np.delete(incomingTracks, track)         # delete index of track in incomming tracks
  indext = np.array(np.nonzero(newTracks != -1), dtype=np.int)[0]   # indexes of assigned tracks
  if indext.size > 0:
    indexp = newTracks[indext]                                    # indexes of assigned peaks
    tfreqn[indext] = pfreqt[indexp]                               # output freq tracks 
    tmagn[indext] = pmagt[indexp]                                 # output mag tracks 
    tphasen[indext] = pphaset[indexp]                             # output phase tracks 
    pfreqt= np.delete(pfreqt, indexp)                             # delete used peaks
    pmagt= np.delete(pmagt, indexp)                               # delete used peaks
    pphaset= np.delete(pphaset, indexp)                           # delete used peaks

  # create new tracks from non used peaks
  emptyt = np.array(np.nonzero(tfreq == 0), dtype=np.int)[0]      # indexes of empty incoming tracks
  peaksleft = np.argsort(-pmagt)                                  # sort left peaks by magnitude
  if ((peaksleft.size > 0) & (emptyt.size >= peaksleft.size)):    # fill empty tracks
      tfreqn[emptyt[:peaksleft.size]] = pfreqt[peaksleft]
      tmagn[emptyt[:peaksleft.size]] = pmagt[peaksleft]
      tphasen[emptyt[:peaksleft.size]] = pphaset[peaksleft]
  elif ((peaksleft.size > 0) & (emptyt.size < peaksleft.size)):   # add more tracks if necessary
      tfreqn[emptyt] = pfreqt[peaksleft[:emptyt.size]]
      tmagn[emptyt] = pmagt[peaksleft[:emptyt.size]]
      tphasen[emptyt] = pphaset[peaksleft[:emptyt.size]]
      tfreqn = np.append(tfreqn, pfreqt[peaksleft[emptyt.size:]])
      tmagn = np.append(tmagn, pmagt[peaksleft[emptyt.size:]])
      tphasen = np.append(tphasen, pphaset[peaksleft[emptyt.size:]])
  
  return tfreqn, tmagn, tphasen

def genSpecSines(ipfreq, ipmag, ipphase, N, fs):
  # Generate a spectrum from a series of sine values
  # ipfreq, ipmag, ipphase: sine peaks frequencies, magnitudes and phases
  # N: size of the complex spectrum to generate
  # fs: sampling frequency
  # returns Y: generated complex spectrum of sines
  Y = UF_C.genSpecSines(N*ipfreq/float(fs), ipmag, ipphase, N)
  return Y

def genSpecSines_p(iploc, ipmag, ipphase, N):
  # Generate a spectrum from a series of sine values
  # iploc, ipmag, ipphase: sine peaks locations, magnitudes and phases
  # N: size of the complex spectrum to generate
  # returns Y: generated complex spectrum of sines
  Y = np.zeros(N, dtype = complex)                 # initialize output spectrum  
  hN = N/2                                         # size of positive freq. spectrum
  for i in range(0, iploc.size):                   # generate all sine spectral lobes
    loc = iploc[i]                                 # it should be in range ]0,hN-1[
    if loc==0 or loc>hN-1: continue
    binremainder = round(loc)-loc;
    lb = np.arange(binremainder-4, binremainder+5) # main lobe (real value) bins to read
    lmag = genBhLobe(lb) * 10**(ipmag[i]/20)  # lobe magnitudes of the complex exponential
    b = np.arange(round(loc)-4, round(loc)+5)
    for m in range(0, 9):
      if b[m] < 0:                                 # peak lobe crosses DC bin
        Y[-b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
      elif b[m] > hN:                              # peak lobe croses Nyquist bin
        Y[b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
      elif b[m] == 0 or b[m] == hN:                # peak lobe in the limits of the spectrum 
        Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i]) + lmag[m]*np.exp(-1j*ipphase[i])
      else:                                        # peak lobe in positive freq. range
        Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i])
    Y[hN+1:] = Y[hN-1:0:-1].conjugate()            # fill the negative part of the spectrum
  return Y

def genBhLobe(x):
  # Generate the transform of the Blackman-Harris window
  # x: bin positions to compute (real values)
  # returns y: transform values
  N = 512;
  f = x*np.pi*2/N                                  # frequency sampling
  df = 2*np.pi/N  
  y = np.zeros(x.size)                               # initialize window
  consts = [0.35875, 0.48829, 0.14128, 0.01168]      # window constants
  for m in range(0,4):  
    y += consts[m]/2 * (D(f-df*m, N) + D(f+df*m, N)) # sum Dirichlet kernels
  y = y/N/consts[0] 
  return y                                           # normalize

def D(x, N):
  # Generate a sinc function (Dirichlet kernel)
  y = np.sin(N * x/2) / np.sin(x/2)
  y[np.isnan(y)] = N                                 # avoid NaN if x == 0
  return y


def peakInterp(mX, pX, ploc):
  # interpolate peak values using parabolic interpolation
  # mX: magnitude spectrum, pX: phase spectrum, ploc: locations of peaks
  # returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
  val = mX[ploc]                                          # magnitude of peak bin 
  lval = mX[ploc-1]                                       # magnitude of bin at left
  rval = mX[ploc+1]                                       # magnitude of bin at right
  iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
  ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
  ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks
  return iploc, ipmag, ipphase

def peakDetection(mX, hN, t):
  # detect spectral peak locations
  # mX: magnitude spectrum, hN: size of positive spectrum, t: threshold
  # returns ploc: peak locations
  thresh = np.where(mX[1:hN-1]>t, mX[1:hN-1], 0);          # locations above threshold
  next_minor = np.where(mX[1:hN-1]>mX[2:], mX[1:hN-1], 0)  # locations higher than the next one
  prev_minor = np.where(mX[1:hN-1]>mX[:hN-2], mX[1:hN-1], 0) # locations higher than the previous one
  ploc = thresh * next_minor * prev_minor                  # locations fulfilling the three criteria
  ploc = ploc.nonzero()[0] + 1
  return ploc

INT16_FAC = (2**15)-1
INT32_FAC = (2**31)-1
INT64_FAC = (2**63)-1

norm_fact = {'int16':INT16_FAC, 'int32':INT32_FAC, 'int64':INT64_FAC,'float32':1.0,'float64':1.0}


def play(y, fs):
  # play the array y as a sound using fs as the sampling rate
  x = copy.deepcopy(y)  #just deepcopying to modify signal to play and to not change original array
  x *= INT16_FAC  #scaling floating point -1 to 1 range signal to int16 range
  x = np.int16(x) #converting to int16 type
  if (fs != 44100): 
    print('WARNING: currently it seems to only work for fs = 44100')
  CHUNK = 1024
  write('temp_file.wav', fs, x)
  wf = wave.open('temp_file.wav', 'rb')
  p = pyaudio.PyAudio()
  stream = p.open(format=p.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
  data = wf.readframes(CHUNK)
  while data is not '':
    stream.write(data)
    data = wf.readframes(CHUNK)
  stream.stop_stream()
  stream.close()
  p.terminate()
  os.remove(os.getcwd()+'/temp_file.wav')
  
def wavread(filename):
  # read a sound file and return an array with the sound and the sampling rate
  (fs, x) = read(filename)
  if len(x.shape) ==2 :
    print "ERROR: Input audio file is stereo. This software only works for mono audio files."
    sys.exit()
    #scaling down and converting audio into floating point number between range -1 to 1
  x = np.float32(x)/norm_fact[x.dtype.name]
  return fs, x
      
def wavwrite(y, fs, filename):
  # write a sound file from an array with the sound and the sampling rate
  x = copy.deepcopy(y)  #just deepcopying to modify signal to write and to not change original array
  x *= INT16_FAC  #scaling floating point -1 to 1 range signal to int16 range
  x = np.int16(x) #converting to int16 type
  write(filename, fs, x)

