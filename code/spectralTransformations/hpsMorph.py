import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import time

import sys, os, functools

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions_C/'))

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

def hpsAnalysis(x, fs, w, wr, pin, N, hN, Ns, hNs, hM, nH, t, f0et, minf0, maxf0, maxhd, stocf):
  
  xw = x[pin-hM:pin+hM-1] * w                                  # window the input sound
  fftbuffer = np.zeros(N)                                      # reset buffer
  fftbuffer[:hM] = xw[hM-1:]                                   # zero-phase window in fftbuffer
  fftbuffer[N-hM+1:] = xw[:hM-1]                           
  X = fft(fftbuffer)                                           # compute FFT
  mX = 20 * np.log10( abs(X[:hN]) )                            # magnitude spectrum of positive frequencies
  ploc = PP.peakDetection(mX, hN, t)                
  pX = np.unwrap( np.angle(X[:hN]) )                           # unwrapped phase spect. of positive freq.     
  iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)            # refine peak values
  
  f0 = fd.f0DetectionTwm(iploc, ipmag, N, fs, f0et, minf0, maxf0)  # find f0
  hloc = np.zeros(nH)                                          # initialize harmonic locations
  hmag = np.zeros(nH)-100                                      # initialize harmonic magnitudes
  hphase = np.zeros(nH)                                        # initialize harmonic phases
  hf = (f0>0) * (f0*np.arange(1, nH+1))                          # initialize harmonic frequencies
  hi = 0                                                       # initialize harmonic index
  npeaks = ploc.size                                           # number of peaks found

  while f0>0 and hi<nH and hf[hi]<fs/2 :                       # find harmonic peaks
    dev = min(abs(iploc/N*fs - hf[hi]))
    pei = np.argmin(abs(iploc/N*fs - hf[hi]))                  # closest peak
    if ( hi==0 or not any(hloc[:hi]==iploc[pei]) ) and dev<maxhd*hf[hi] :
      hloc[hi] = iploc[pei]                                    # harmonic locations
      hmag[hi] = ipmag[pei]                                    # harmonic magnitudes
      hphase[hi] = ipphase[pei]                                # harmonic phases
    hi += 1                                                    # increase harmonic index
    
  hloc = (hloc!=0) * (hloc*Ns/N)                               # synth. locs
  ri = pin-hNs-1                                               # input sound pointer for residual analysis
  xw2 = x[ri:ri+Ns]*wr                                         # window the input sound                                       
  fftbuffer = np.zeros(Ns)                                     # reset buffer
  fftbuffer[:hNs] = xw2[hNs:]                                  # zero-phase window in fftbuffer
  fftbuffer[hNs:] = xw2[:hNs]                            
  X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
  
  Xh = GS.genSpecSines(hloc, hmag, hphase, Ns)                    # generate sines
  Xr = X2-Xh                                                   # get the residual complex spectrum
  mXr = 20 * np.log10( abs(Xr[:hNs]) )                         # magnitude spectrum of residual
  mXrenv = resample(np.maximum(-200, mXr), mXr.size*stocf)         # decimate the magnitude spectrum and avoid -Inf    

  return f0, hloc, hmag, mXrenv


def hpsMorph(x, x2, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, f0intp, htintp, rintp):
  # morph between two sounds using the harmonic plus stochastic model
  # x,x2: input sounds, fs: sampling rate, w: analysis window (odd size), 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
  # maxf0: maximim f0 frequency in Hz, 
  # f0et: error threshold in the f0 detection (ex: 5),
  # maxhd: max. relative deviation in harmonic detection (ex: .2)
  # stocf: decimation factor of mag spectrum for stochastic analysis
  # f0intp: f0 interpolation factor,
  # htintp: harmonic timbre interpolation factor,
  # rintp: residual interpolation factor,
  # y: output sound, yh: harmonic component, ys: stochastic component
  
  if isinstance(f0intp, int) :
    in_time = np.array([0, np.float(x.size)/fs])    #input time
    control = np.array([f0intp, f0intp])   #control value
    f0intp = np.asarray((in_time, control))
  
  if isinstance(htintp, int) :
    in_time = np.array([0, np.float(x.size)/fs])    #input time
    control = np.array([htintp, htintp])   #control value
    htintp = np.asarray((in_time, control))
      
  if isinstance(rintp, int) :
    in_time = np.array([0, np.float(x.size)/fs])    #input time
    control = np.array([rintp, rintp])   #control value
    rintp = np.asarray((in_time, control))

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
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
  sws = H*hanning(Ns)/2                                         # synthesis window for stochastic
  lastyhloc = np.zeros(nH)                                      # initialize synthesis harmonic locations
  yhphase = 2*np.pi * np.random.rand(nH)                        # initialize synthesis harmonic phases
  minpin2 = max(hNs, hM)                                        # minimum sample value for x2
  maxpin2 = x2.size-max(hNs, hM)                                # maximum sample value for x2

  while pin<pend :       
  
  #-----analysis----- 
    
    #-----first sound analysis-----
    f0, hloc, hmag, mXrenv = hpsAnalysis(x,fs,w,wr,pin,N,hN,Ns,hNs,hM,nH,t,f0et,minf0,maxf0,maxhd,stocf)
    
    #-----second sound analysis-----
    pin2 = round(np.float(pin)/x.size*x2.size) # linear time mapping between inputs
    pin2 = min(maxpin2, max(minpin2,pin2))
    f02, hloc2, hmag2, mXrenv2 = hpsAnalysis(x2,fs,w,wr,pin2,N,hN,Ns,hNs,hM,nH,t,f0et,minf0,maxf0,maxhd,stocf)

  #-----morph-----
    cf0intp = np.interp(np.float(pin)/fs, f0intp[0,:], f0intp[1,:])   # get control value 
    chtintp = np.interp(np.float(pin)/fs, htintp[0,:], htintp[1,:])   # get control value
    crintp = np.interp(np.float(pin)/fs, rintp[0,:], rintp[1,:])      # get control value
    
    if f0>0 and f02>0 :
      yf0 = f0*(1-cf0intp) + f02*cf0intp              # both inputs are harmonic 
      yhloc = np.arange(1, nH+1)*yf0/fs*Ns            # generate synthesis harmonic serie
      
      idx = np.where((hloc>0) & (hloc<hNs))
      xp = np.concatenate( (np.array([0]), hloc[idx], np.array([Ns])) )
      fp = np.concatenate( (np.array([hmag[0]]), hmag[idx], np.array([hmag[-1]])) )
      yhmag = np.interp(yhloc, xp, fp)                # interpolated envelope
      
      ####### MAYBE WE CAN IMPROVE IT USING DIRECTLY THIS AS INDEX: [(hloc2>0) & (hloc2<hNs)]
      idx2 = np.where((hloc2>0) & (hloc2<hNs)) 
      xp = np.concatenate( (np.array([0]), hloc2[idx2], np.array([Ns])) )
      fp = np.concatenate( (np.array([hmag2[0]]), hmag2[idx2], np.array([hmag2[-1]])) )
      yhmag2 = np.interp(yhloc, xp, fp)               # interpolated envelope
      yhmag = yhmag*(1-chtintp) + yhmag2*chtintp      # timbre morphing
    
    else :
      yf0 = 0                                         # remove harmonic content
      yhloc = hloc*0
      yhmag = hmag*0

    mYrenv = mXrenv*(1-crintp) + mXrenv2*crintp 

  #-----synthesis-----
    yhphase += 2*np.pi * (lastyhloc+yhloc)/2/Ns*H                 # propagate phases
    lastyhloc = yhloc 
    
    Yh = GS.genSpecSines(yhloc, yhmag, yhphase, Ns)                  # generate spec sines 
    mYs = resample(mYrenv, hNs)                                   # interpolate to original size
    pYs = 2*np.pi * np.random.rand(hNs)                           # generate phase random values
    
    Ys = np.zeros(Ns, dtype = complex)
    Ys[:hNs] = 10**(mYs/20) * np.exp(1j*pYs)                      # generate positive freq.
    Ys[hNs+1:] = 10**(mYs[:0:-1]/20) * np.exp(-1j*pYs[:0:-1])     # generate negative freq.

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Yh) )                            
    yhw[:hNs-1] = fftbuffer[hNs+1:]                               # sines in time domain using IFFT
    yhw[hNs-1:] = fftbuffer[:hNs+1] 
    
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real( ifft(Ys) )
    ysw[:hNs-1] = fftbuffer[hNs+1:]                               # stochastic in time domain using IFFT
    ysw[hNs-1:] = fftbuffer[:hNs+1]

    ro = pin-hNs
    yh[ro:ro+Ns] += sw*yhw                                        # overlap-add for sines
    ys[ro:ro+Ns] += sws*ysw                                       # overlap-add for stoch
    pin += H                                                      # advance sound pointer
    
  y = yh+ys
  return y, yh, ys



def defaultTest():
    
    str_time = time.time()
    fs, x = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/soprano-E4.wav'))
    fs, x2 = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/violin-B3.wav'))
    
    
    w = np.hamming(1025)
    N = 2048
    t = -150
    nH = 200
    minf0 = 100
    maxf0 = 400
    f0et = 5
    maxhd = 0.2
    stocf = 0.1
    dur = x.size/fs
    f0intp = np.array([[ 0, dur], [0, 1]])
    htintp = np.array([[ 0, dur], [0, 1]]) 
    rintp = np.array([[ 0, dur], [0, 1]])
    y, yh, ys = hpsMorph(x, x2, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, f0intp, htintp, rintp)

    print "time taken for computation " + str(time.time()-str_time)
  
if __name__ == '__main__':
      
    fs, x = wp.wavread('../../sounds/soprano-E4.wav')
    fs, x2 = wp.wavread('../../sounds/violin-B3.wav')

    w = np.hamming(1025)
    N = 2048
    t = -150
    nH = 200
    minf0 = 100
    maxf0 = 400
    f0et = 5
    maxhd = 0.2
    stocf = 0.1
    dur = x.size/fs
    f0intp = np.array([[ 0, dur], [0, 1]])
    htintp = np.array([[ 0, dur], [0, 1]]) 
    rintp = np.array([[ 0, dur], [0, 1]])
    y, yh, ys = hpsMorph(x, x2, fs, w, N, t, nH, minf0, maxf0, f0et, maxhd, stocf, f0intp, htintp, rintp)

    wp.play(y, fs)
    wp.play(yh, fs)
    wp.play(ys, fs)