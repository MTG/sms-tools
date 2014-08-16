# functions that implement analysis and synthesis of sounds using the Harmonic Model
# (for example usage check the models_interface directory)

import numpy as np
from scipy.signal import blackmanharris, triang
import math
import dftModel as DFT
import utilFunctions as UF
import sineModel as SM

def f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et):
  # Fundamental frequency detection using twm algorithm
  # x: input sound; fs: sampling rate; w: analysis window; 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # minf0: minimum f0 frequency in Hz, maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # returns f0: fundamental frequency

  hN = N/2                                        # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))             # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                 # half analysis window size by floor
  x = np.append(np.zeros(hM2),x)                  # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hM1))                  # add zeros at the end to analyze last sample
  pin = hM1                                       # init sound pointer in middle of anal window          
  pend = x.size - hM1                             # last sample to start a frame
  fftbuffer = np.zeros(N)                         # initialize buffer for FFT
  w = w / sum(w)                                  # normalize analysis window
  f0 = []
  f0t = 0
  f0stable = 0
  while pin<pend:             
    x1 = x[pin-hM1:pin+hM2]                       # select frame
    mX, pX = DFT.dftAnal(x1, w, N)                # compute dft           
    ploc = UF.peakDetection(mX, t)                # detect peak locations   
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
    if ((f0stable==0)&(f0t>0)) \
        or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
      f0stable = f0t                                # consider a stable f0 if it is close to the previous one
    else:
      f0stable = 0

    f0 = np.append(f0, f0t)
    pin += H                                        # advance sound pointer
  return f0

def harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et):
  # Analysis/synthesis of a sound using the sinusoidal harmonic model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # returns y: output array sound

  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hM1))                          # add zeros at the end to analyze last sample
  Ns = 512                                                # FFT size for synthesis (even)
  H = Ns/4                                                # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window          
  pend = x.size - max(hNs, hM1)                           # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yh = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window
  sw = np.zeros(Ns)                                       # initialize synthesis window
  ow = triang(2*H)                                        # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                 # synthesis window
  bh = bh / sum(bh)                                       # normalize synthesis window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # window for overlap-add
  hfreqp = []
  f0t = 0
  f0stable = 0
  while pin<pend:             
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
    ploc = UF.peakDetection(mX, t)                        # detect peak locations     
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
    if ((f0stable==0)&(f0t>0)) \
        or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
      f0stable = f0t                                # consider a stable f0 if it is close to the previous one
    else:
      f0stable = 0
    hfreq, hmag, hphase = UF.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs) # find harmonics
    hfreqp = hfreq
  #-----synthesis-----
    Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)     # generate spec sines          
    fftbuffer = np.real(ifft(Yh))                         # inverse FFT
    yh[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yh[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yh                           # overlap-add
    pin += H                                              # advance sound pointer
  y = np.delete(y, range(hM2))                   # delete half of first window which was added in stftAnal
  y = np.delete(y, range(y.size-hM1, y.size))    # add zeros at the end to analyze last sample
  return y

def harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope=0.01, minSineDur=.02):
  # Analysis of a sound using the sinusoidal harmonic model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512); t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
  # harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
  # returns xhfreq, xhmag, xhphase: harmonic frequencies, magnitudes and phases

  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
  pin = hM1                                               # init sound pointer in middle of anal window          
  pend = x.size - hM1                                     # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  w = w / sum(w)                                          # normalize analysis window
  hfreqp = []
  f0t = 0
  f0stable = 0
  while pin<=pend:           
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft            
    ploc = UF.peakDetection(mX, t)                        # detect peak locations   
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
    if ((f0stable==0)&(f0t>0)) \
        or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
      f0stable = f0t                                # consider a stable f0 if it is close to the previous one
    else:
      f0stable = 0
    hfreq, hmag, hphase = UF.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs, harmDevSlope) # find harmonics
    hfreqp = hfreq
    if pin == hM1: 
      xhfreq = np.array([hfreq])
      xhmag = np.array([hmag])
      xhphase = np.array([hphase])
    else:
      xhfreq = np.vstack((xhfreq,np.array([hfreq])))
      xhmag = np.vstack((xhmag, np.array([hmag])))
      xhphase = np.vstack((xhphase, np.array([hphase])))
    pin += H                                              # advance sound pointer
  xhfreq = UF.cleaningSineTracks(xhfreq, round(fs*minSineDur/H))
  return xhfreq, xhmag, xhphase

