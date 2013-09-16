import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import time

import sys, os, functools

sys.path.append(os.path.realpath('../UtilityFunctions/'))
sys.path.append(os.path.realpath('../UtilityFunctions_C/'))
import sms_f0detectiontwm as fd
import sms_wavplayer as wp
import sms_PeakProcessing as PP

try:
  import UtilityFunctions_C as GS
except ImportError:
  import sms_GenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

def sps(x, fs, w, N, t, maxnS, stocf) :
  # Analysis/synthesis of a sound using the sinusoidal plus stochastic model
  # x: input sound, fs: sampling rate, w: analysis window (odd size), 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # maxnS: maximum number of sinusoids,
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # y: output sound, yh: harmonic component, ys: stochastic component

  hN = N/2                                                      # size of positive spectrum
  hM = (w.size+1)/2                                             # half analysis window size
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM)                                            # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM)                                  # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  yhw = np.zeros(Ns)                                            # initialize output sine sound frame
  ysw = np.zeros(Ns)                                            # initialize output residual sound frame
  yh = np.zeros(x.size)                                         # initialize output sine component
  ys = np.zeros(x.size)                                         # initialize output residual component
  x = np.float32(x) / (2**15)                                   # normalize input signal
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  sws = H*hanning(Ns)/2                                         # synthesis window for stochastic
  lastysloc = np.zeros(maxnS)                                   # initialize synthesis harmonic locations
  ysphase = 2*np.pi * np.random.rand(maxnS)                     # initialize synthesis harmonic phases
  fridx = 0                                                     # frame pointer
  isInitFrame = True                                            # True for frames equivalent to initial frame (for synth part)
  lastnS = 0                                                    # it doesnot harm to initialize this variable with 0.

  while pin<pend:       
    
    if fridx==0 or lastnS==0 :     # whenever lastnS is zero implies frame is equivalent to initial frame
      isInitFrame = True

  #-----analysis-----             
    xw = x[pin-hM:pin+hM-1] * w                                  # window the input sound
    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM] = xw[hM-1:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]              
    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
    ploc = PP.peak_detection(mX, hN, t)                
    pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.     
    iploc, ipmag, ipphase = PP.peak_interp(mX, pX, ploc)            # refine peak values
    
    smag = np.sort(ipmag)[::-1]                                  # sort peaks by magnitude in descending order
    I = np.argsort(ipmag)[::-1]
    
    nS = min(maxnS, np.where(smag>t)[0].size)                    # get peaks above threshold
    sloc = iploc[I[:nS]]
    sphase = ipphase[I[:nS]]  
    if isInitFrame :                                             # update last frame data
      lastnS = nS
      lastsloc = sloc
      lastsmag = smag
      lastsphase = sphase

    sloc = (sloc!=0) * (sloc*Ns/N)                               # peak locations for synthesis
    lastidx = np.zeros(nS, dtype = int)
    for i in range(0, nS) :  # find closest peak to create trajectories
      idx = np.argmin(abs(sloc[i] - lastsloc[:lastnS]))  
      lastidx[i] = idx

    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                         # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                  # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]              
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
    
    Xh = GS.genspecsines(sloc, smag, sphase, Ns)                    # generate sines
    Xr = X2-Xh                                                   # get the residual complex spectrum
    mXr = 20 * np.log10( abs(Xr[:hNs]) )                         # magnitude spectrum of residual
    mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)     # decimate the magnitude spectrum and avoid -Inf                     

  #-----synthesis data-----
    ysloc = sloc                                                 # synthesis harmonics locs
    ysmag = smag[:nS]                                            # synthesis harmonic amplitudes
    mYrenv = mXrenv                                              # synthesis residual envelope

  #-----transformations-----

    #----- filtering -----
    # Hz = np.array([0, 2099, 2100, 3000, 3001, fs/2]) # Hz
    # dB = np.array([-200, -200, 0, 0, -200, -200])    # dB
    # Filter = np.asarray((Hz, dB))
    # ysmag += np.interp(ysloc/Ns*fs, Filter[0,:],Filter[1,:])

    #-----frequency shift-----
    # fshift = 100
    # ysloc = (ysloc>0) * (ysloc + fshift/fs*Ns)                 # frequency shift in Hz
     
    #-----frequency stretch-----
    # fstretch = 1.1
    # ysloc = ysloc * (fstretch**np.arange(0, ysloc.size))
     
    #-----frequency scale-----
    # fscale = 1.2
    # ysloc = ysloc * fscale

  #-----synthesis-----
    
    if isInitFrame :
      # Variables need to be initialized like for the first frame
      lastysloc = np.zeros(maxnS)                     # initialize synthesis harmonic locations
      ysphase = 2*np.pi * np.random.rand(maxnS)       # initialize synthesis harmonic phases
      
      lastysphase = ysphase                           # phase for first frame
    
    if nS>lastnS :                                    # initialize peaks that start
      lastysphase = np.concatenate((lastysphase, np.zeros(nS-lastnS)))
      lastysloc = np.concatenate((lastysloc, np.zeros(nS-lastnS)))
    
    ysphase = lastysphase[lastidx] + 2*np.pi*(lastysloc[lastidx]+ysloc)/2/Ns*H # propagate phases
    
    lastysloc = ysloc
    lastysphase = ysphase  
    lastnS = nS                                       # update last frame data
    lastsloc = sloc                                   # update last frame data
    lastsmag = smag                                   # update last frame data
    lastsphase = sphase                               # update last frame data

    Yh = GS.genspecsines(ysloc, ysmag, ysphase, Ns)      # generate spec sines 
    mYs = resample(mYrenv, hNs)                       # interpolate to original size
    pYs = 2*np.pi*np.random.rand(hNs)                 # generate phase random values
    
    Ys = np.zeros(Ns, dtype = complex)
    Ys[:hNs] = 10**(mYs/20) * np.exp(1j*pYs)                   # generate positive freq.
    Ys[hNs+1:] = 10**(mYs[:0:-1]/20) * np.exp(-1j*pYs[:0:-1])  # generate negative freq.

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Yh) )                            
    yhw[:hNs-1] = fftbuffer[hNs+1:]                   # sines in time domain using IFFT
    yhw[hNs-1:] = fftbuffer[:hNs+1] 
    
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Ys) )
    ysw[:hNs-1] = fftbuffer[hNs+1:]                   # stochastic in time domain using IFFT
    ysw[hNs-1:] = fftbuffer[:hNs+1]

    yh[ri:ri+Ns] += sw*yhw                            # overlap-add for sines
    ys[ri:ri+Ns] += sws*ysw                           # overlap-add for stoch
    pin += H                                          # advance sound pointer
    fridx += 1                                        # advance frame pointer
    isInitFrame = False                               # variable meaningful only for current frame,
                                                      # therefore False at each frame
  y = yh+ys
  return y, yh, ys

def DefaultTest():
  
  str_time = time.time()
    
  (fs, x) = read('../../sounds/speech-female.wav')
  # wp.play(x, fs)

  # fig = plt.figure()
  w = np.hamming(801)
  N = 1024
  t = -120
  maxnS = 30
  stocf = 0.5
  y, yh, ys = sps(x, fs, w, N, t, maxnS, stocf)

  y *= 2**15
  y = y.astype(np.int16)

  yh *= 2**15
  yh = yh.astype(np.int16)

  ys *= 2**15
  ys = ys.astype(np.int16)
  
  print "time taken for computation " + str(time.time()-str_time)  
  

if __name__ == '__main__':

  (fs, x) = read('../../sounds/speech-female.wav')
  # wp.play(x, fs)

  # fig = plt.figure()
  w = np.hamming(801)
  N = 1024
  t = -120
  maxnS = 30
  stocf = 0.5
  y, yh, ys = sps(x, fs, w, N, t, maxnS, stocf)

  y *= 2**15
  y = y.astype(np.int16)

  yh *= 2**15
  yh = yh.astype(np.int16)

  ys *= 2**15
  ys = ys.astype(np.int16)

  wp.play(y, fs)
  wp.play(yh, fs)
  wp.play(ys, fs)