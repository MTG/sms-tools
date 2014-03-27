import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time
import harmonicModel as HM
import stft as STFT
import dftModel as DFT
import utilFunctions as UF
  
def hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et):
  # Analysis/synthesis of a sound using the harmonic plus residual model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # returns y: output sound, yh: harmonic component, xr: residual component

  hN = N/2                                                      # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  yhw = np.zeros(Ns)                                            # initialize output sound frame
  xrw = np.zeros(Ns)                                            # initialize output sound frame
  yh = np.zeros(x.size)                                         # initialize output array
  xr = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  hfreqp = []
  f0t = 0
  f0stable = 0
  while pin<pend:  
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                          # select frame
    mX, pX = DFT.dftAnal(x1, w, N)                    # compute dft
    ploc = UF.peakDetection(mX, hN, t)               # find peaks 
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values
    ipfreq = fs * iploc/N
    f0t = UF.f0DetectionTwm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
    if ((f0stable==0)&(f0t>0)) \
      or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
      f0stable = f0t                                # consider a stable f0 if it is close to the previous one
    else:
      f0stable = 0
    hfreq, hmag, hphase = UF.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs) # find harmonics
    hfreqp = hfreq
    ri = pin-hNs-1                                  # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                            # window the input sound                     
    fftbuffer = np.zeros(Ns)                        # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                     # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]                     
    X2 = fft(fftbuffer)                             # compute FFT for residual analysis
    #-----synthesis-----
    Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)    # generate sines
    Xr = X2-Yh                                      # get the residual complex spectrum                       
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Yh))                   # inverse FFT of harmonic spectrum
    yhw[:hNs-1] = fftbuffer[hNs+1:]                 # undo zero-phase window
    yhw[hNs-1:] = fftbuffer[:hNs+1] 
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Xr))                   # inverse FFT of residual spectrum
    xrw[:hNs-1] = fftbuffer[hNs+1:]                 # undo zero-phase window
    xrw[hNs-1:] = fftbuffer[:hNs+1]
    yh[ri:ri+Ns] += sw*yhw                          # overlap-add for sines
    xr[ri:ri+Ns] += sw*xrw                          # overlap-add for residual
    pin += H                                        # advance sound pointer
  y = yh+xr                                         # sum of harmonic and residual components
  return y, yh, xr
 
if __name__ == '__main__':
  (fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/sax-phrase.wav'))
  w = np.blackman(601)
  N = 1024
  t = -100
  nH = 100
  minf0 = 350
  maxf0 = 700
  f0et = 5
  maxnpeaksTwm = 5
  minSineDur = .1
  harmDevSlope = 0.01
  Ns = 512
  H = Ns/4
  hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
  xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
  mXr, pXr = STFT.stftAnal(xr, fs, hamming(Ns), Ns, H)
  yh = HM.harmonicModelSynth(hfreq, hmag, hphase, Ns, H, fs)

  UF.play(yh, fs)
  UF.play(xr, fs)


  plt.figure(1, figsize=(9.5, 7))
  maxplotfreq = 20000.0
  numFrames = int(mXr[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs)                             
  binFreq = fs*np.arange(Ns*maxplotfreq/fs)/Ns                        
  plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:,:Ns*maxplotfreq/fs+1]))
  plt.autoscale(tight=True)

  harms = hfreq*np.less(hfreq,maxplotfreq)
  harms[harms==0] = np.nan
  numFrames = int(harms[:,0].size)
  frmTime = H*np.arange(numFrames)/float(fs) 
  plt.plot(frmTime, harms, color='k', ms=3, alpha=1)
  plt.xlabel('Time(s)')
  plt.ylabel('Frequency(Hz)')
  plt.autoscale(tight=True)
  plt.title('harmonic + residual components')

  plt.tight_layout()
  plt.show()
