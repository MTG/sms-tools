import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import time

import sys, os

sys.path.append(os.path.realpath('../basicFunctions/'))
sys.path.append(os.path.realpath('../basicFunctions_C/'))
import smsF0DetectionTwm as fd
import smsWavplayer as wp
import smsPeakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  
# def extrap(x, xp, yp):
#   # np.interp function with linear extrapolation

#   y = np.interp(x, xp, yp)
#   if x < xp[0] :
#     y = yp[0] + (x-xp[0]) * (yp[0]-yp[1]) / (xp[0]-xp[1])
#   elif x > xp[-1] :
#     y = yp[-1] + (x-xp[-1]) * (yp[-1]-yp[-2])/(xp[-1]-xp[-2])
  
#   return y

def sps_timescale(x, fs, w, N, t, maxnS, stocf) :
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
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  tm = np.arange(0.01, 0.94, 0.01)
  in_time = np.concatenate( (np.array([0]), tm+0.05*np.sin(8.6*np.pi*tm), np.array([1])) ) # input time --> keep end value
  out_time = np.concatenate( (np.array([0]),                 tm           , np.array([1])) ) # output time
  timemapping = np.asarray( (in_time, out_time) ) 
  # timemapping = np.array( [[0, 1], [0, 2]] )                  # input time (sec), output time (sec)                      
  timemapping = timemapping * x.size/fs
  outsoundlength = round(timemapping[1, -1]*fs)                 # length of output sound
  pend = outsoundlength - max(hNs, hM)                          # last sample to start a frame
  yhw = np.zeros(Ns)                                            # initialize output sine sound frame
  ysw = np.zeros(Ns)                                            # initialize output residual sound frame
  yh = np.zeros(outsoundlength)                                 # initialize output sine component
  ys = np.zeros(outsoundlength)                                 # initialize output residual component
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
  ysphase = 2*np.pi * np.random.rand(maxnS)                   # initialize synthesis harmonic phases

  minpin = max(hNs, hM)
  maxpin = x.size - max(hNs,hM)
  fridx = 0                                                     # frame pointer
  isInitFrame = True                                            # True for frames equivalent to initial frame (for synth part)
  lastnS = 0                                                    # it doesnot harm to initialize this variable with 0.
  pout = pin
  
  while pout<pend:       
    
    if fridx==0 or lastnS==0 :     # whenever lastnS is zero implies frame is equivalent to initial frame
      isInitFrame = True
    
    pin = round(np.interp(np.float(pout)/fs, timemapping[1,:],timemapping[0,:]) * fs )
    pin = max(minpin, pin)
    pin = min(maxpin, pin)
  
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
    for i in range(0, nS) :                                      # find closest peak to create trajectories
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
  
  #-----synthesis-----
    
    if isInitFrame :
      # Variables need to be initialized like for the first frame
      lastysloc = np.zeros(maxnS)                     # initialize synthesis harmonic locations
      ysphase = 2*np.pi * np.random.rand(maxnS)     # initialize synthesis harmonic phases
      
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
    pYs = 2*np.pi*np.random.rand(hNs)               # generate phase random values
    
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

    ro = pout-hNs                                     # output sound pointer for overlap
    yh[ro:ro+Ns] += sw*yhw                            # overlap-add for sines
    ys[ro:ro+Ns] += sws*ysw                           # overlap-add for stochastic
    pout += H                                         # advance sound pointer
    fridx += 1                                        # advance frame pointer
    isInitFrame = False                               # variable meaningful only for current frame,
                                                      # therefore False at each frame
  y = yh+ys
  return y, yh, ys

def DefaultTest():
    
    str_time = time.time()
	  
    (fs, x) = wp.wavread('../../sounds/speech-female.wav')
    w = np.hamming(801)
    N = 1024
    t = -120
    maxnS = 30
    stocf = 0.5
    y, yh, ys = sps_timescale(x, fs, w, N, t, maxnS, stocf)

    print "time taken for computation " + str(time.time()-str_time)
  
if __name__ == '__main__':     
	  
    (fs, x) = wp.wavread('../../sounds/speech-female.wav')
    # wp.play(x, fs)

    w = np.hamming(801)
    N = 1024
    t = -120
    maxnS = 30
    stocf = 0.5
    y, yh, ys = sps_timescale(x, fs, w, N, t, maxnS, stocf)

    wp.play(y, fs)
    wp.play(yh, fs)
    wp.play(ys, fs)