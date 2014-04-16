import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
import dftModel as DFT
import stft as STFT
import utilFunctions as UF

  
def sineModel(x, fs, w, N, t):
  # Analysis/synthesis of a sound using the sinusoidal model
  # x: input array sound, w: analysis window, N: size of complex spectrum,
  # t: threshold in negative dB 
  # returns y: output array sound
  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  Ns = 512                                                # FFT size for synthesis (even)
  H = Ns/4                                                # Hop size used for analysis and synthesis
  hNs = Ns/2                                              # half of synthesis FFT size
  pin = max(hNs, hM1)                                     # init sound pointer in middle of anal window       
  pend = x.size - max(hNs, hM1)                           # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yw = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window
  sw = np.zeros(Ns)                                       # initialize synthesis window
  ow = triang(2*H);                                       # triangular window
  sw[hNs-H:hNs+H] = ow                                    # add triangular window
  bh = blackmanharris(Ns)                                 # blackmanharris window
  bh = bh / sum(bh)                                       # normalized blackmanharris window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window
  while pin<pend:                                         # while input sound pointer is within sound 
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
    ploc = UF.peakDetection(mX, hN, t)                    # detect locations of peaks
    pmag = mX[ploc]                                       # get the magnitude of the peaks
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
  #-----synthesis-----
    plocs = iploc*Ns/N                                    # adapt peak locations to size of synthesis FFT
    Y = UF.genSpecSines(fs*plocs/N, ipmag, ipphase, Ns, fs)    # generate sines in the spectrum         
    fftbuffer = np.real(ifft(Y))                          # compute inverse FFT
    yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yw[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
    pin += H                                              # advance sound pointer
  return y

def sineModelAnal(x, fs, w, N, H, t, maxnSines = 100, minSineDur=.01, freqDevOffset=20, freqDevSlope=0.01):
  # Analysis of a sound using the sinusoidal model
  # x: input array sound, w: analysis window, N: size of complex spectrum,
  # H: hop-size, t: threshold in negative dB
  # maxnSines: maximum number of sines per frame
  # minSineDur: minimum duration of sines in seconds
  # freqDevOffset: minimum frequency deviation at 0Hz 
  # freqDevSlope: slope increase of minimum frequency deviation
  # returns xtfreq, xtmag, xtphase: frequencies, magnitudes and phases of sinusoids
  hN = N/2                                                # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  x = np.append(np.zeros(hM2),x)                          # add zeros at beginning to center first window at sample 0
  x = np.append(x,np.zeros(hM2))                          # add zeros at the end to analyze last sample
  pin = hM1                                               # initialize sound pointer in middle of analysis window       
  pend = x.size - hM1                                     # last sample to start a frame
  w = w / sum(w)                                          # normalize analysis window
  tfreq = np.array([])
  while pin<pend:                                         # while input sound pointer is within sound            
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = DFT.dftAnal(x1, w, N)                        # compute dft
    ploc = UF.peakDetection(mX, hN, t)                    # detect locations of peaks
    pmag = mX[ploc]                                       # get the magnitude of the peaks
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)   # refine peak values by interpolation
    ipfreq = fs*iploc/float(N)
    tfreq, tmag, tphase = UF.sineTracking(ipfreq, ipmag, ipphase, tfreq, freqDevOffset, freqDevSlope)
    tfreq = np.resize(tfreq, min(maxnSines, tfreq.size))
    tmag = np.resize(tmag, min(maxnSines, tmag.size))
    tphase = np.resize(tphase, min(maxnSines, tphase.size))
    jtfreq = np.zeros(maxnSines) 
    jtmag = np.zeros(maxnSines)  
    jtphase = np.zeros(maxnSines)    
    jtfreq[:tfreq.size]=tfreq 
    jtmag[:tmag.size]=tmag
    jtphase[:tphase.size]=tphase 
    if pin == hM1:
      xtfreq = jtfreq 
      xtmag = jtmag
      xtphase = jtphase
    else:
      xtfreq = np.vstack((xtfreq, jtfreq))
      xtmag = np.vstack((xtmag, jtmag))
      xtphase = np.vstack((xtphase, jtphase))
    pin += H
  xtfreq = UF.cleaningSineTracks(xtfreq, round(fs*minSineDur/H))
  return xtfreq, xtmag, xtphase


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
    phase += (np.pi*(lastfreq+freq[l])/fs)*H            # propagate phases
    Y = UF.genSpecSines(freq[l], mag[l], phase, N, fs)    # generate sines in the spectrum         
    lastfreq = freq[l]                                    # save frequency for phase propagation
    yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
    y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
    pout += H                                             # advance sound pointer
  y = np.delete(y, range(hN))                             # delete half of first window
  y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window 
  return y

def sineModelSynth(tfreq, tmag, tphase, N, H, fs):
  # Synthesis of a sound using the sinusoidal model
  # tfreq,tmag, tphase: frequencies, magnitudes and phases of sinusoids,
  # N: synthesis FFT size, H: hop size, fs: sampling rate
  # returns y: output array sound
  hN = N/2                                                # half of FFT size for synthesis
  L = tfreq[:,0].size                                     # number of frames
  pout = 0                                                # initialize output sound pointer         
  ysize = H*(L+3)                                         # output sound size
  y = np.zeros(ysize)                                     # initialize output array
  sw = np.zeros(N)                                        # initialize synthesis window
  ow = triang(2*H);                                       # triangular window
  sw[hN-H:hN+H] = ow                                      # add triangular window
  bh = blackmanharris(N)                                  # blackmanharris window
  bh = bh / sum(bh)                                       # normalized blackmanharris window
  sw[hN-H:hN+H] = sw[hN-H:hN+H]/bh[hN-H:hN+H]             # normalized synthesis window
  lastytfreq = tfreq[0,:]                                 # initialize synthesis frequencies
  ytphase = 2*np.pi*np.random.rand(tfreq[0,:].size)       # initialize synthesis phases 
  for l in range(L):                                      # iterate over all frames
    if (tphase.size > 0):                                 # if no phases generate them
      ytphase = tphase[l,:] 
    else:
      ytphase += (np.pi*(lastytfreq+tfreq[l,:])/fs)*H     # propagate phases
    Y = UF.genSpecSines(tfreq[l,:], tmag[l,:], ytphase, N, fs)  # generate sines in the spectrum         
    lastytfreq = tfreq[l,:]                               # save frequency for phase propagation
    yw = np.real(fftshift(ifft(Y)))                       # compute inverse FFT
    y[pout:pout+N] += sw*yw                               # overlap-add and apply a synthesis window
    pout += H                                             # advance sound pointer
  y = np.delete(y, range(hN))                             # delete half of first window
  y = np.delete(y, range(y.size-hN, y.size))              # delete half of the last window 
  return y
  
# test sineModelAnal and sineModelSynth
if __name__ == '__main__':
  (fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/bendir.wav'))
  w = np.hamming(2001)
  N = 2048
  H = 128
  t = -80
  minSineDur = .02
  maxnSines = 150
  freqDevOffset = 10
  freqDevSlope = 0.001
  mX, pX = STFT.stftAnal(x, fs, w, N, H)
  tfreq, tmag, tphase = sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

  plt.figure(1, figsize=(9.5, 7))
  maxplotfreq = 5000.0
  maxplotbin = int(N*maxplotfreq/fs)
  numFrames = int(mX[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = np.arange(maxplotbin+1)*float(fs)/N                         
  plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:maxplotbin+1]))
  plt.autoscale(tight=True)
  
  tracks = tfreq*np.less(tfreq, maxplotfreq)
  tracks[tracks<=0] = np.nan
  plt.plot(frmTime, tracks, color='k')
  plt.autoscale(tight=True)
  plt.title('sinusoidal tracks on spectrogram')

  Ns = 512
  y = sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)
  UF.play(y, fs)

  plt.tight_layout()
  plt.show()
