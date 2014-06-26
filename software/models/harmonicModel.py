import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
import dftModel as DFT
import stft as STFT
import utilFunctions as UF
import sineModel as SM

def f0Twm(x, fs, w, N, H, t, minf0, maxf0, f0et):
  # fundamental frequency detection using twm algorithm
  # x: input sound, fs: sampling rate, w: analysis window, 
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
    ploc = UF.peakDetection(mX, hN, t)            # detect peak locations   
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0t = UF.f0DetectionTwm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
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
  x = np.append(np.zeros(hM2),x)                 # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hM1))                 # add zeros at the end to analyze last sample
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
    mX, pX = DFT.dftAnal(x1, w, N)                    # compute dft
    ploc = UF.peakDetection(mX, hN, t)                    # detect peak locations     
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0t = UF.f0DetectionTwm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
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
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # harmDevSlope: slope of harmonic deviation
  # minSineDur: minimum length of harmonics
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
    ploc = UF.peakDetection(mX, hN, t)                    # detect peak locations   
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values
    ipfreq = fs * iploc/N
    f0t = UF.f0DetectionTwm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
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


# example of using harmonicModelAnal and harmonicModelSynth
if __name__ == '__main__':

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
  hfreq, hmag, hphase = harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)

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

