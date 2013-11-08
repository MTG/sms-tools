import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift
import math
import sys, os, functools, time

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions_C/'))

import smsWavplayer as wp
import smsPeakProcessing as PP

try:
  import basicFunctions_C as GS
except ImportError:
  import smsGenSpecSines as GS
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  print "NOTE: Cython modules for some functions were not imported, the processing will be slow"
  print "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%"
  

  
def sprModel(x, fs, w, N, t):
  # Analysis/synthesis of a sound using the sinusoidal plus residual model
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # y: output sound, ys: sinusoidal component, yr: residual component

  hN = N/2                                                      # size of positive spectrum
  hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
  Ns = 512                                                      # FFT size for synthesis (even)
  H = Ns/4                                                      # Hop size used for analysis and synthesis
  hNs = Ns/2      
  pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window          
  pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
  fftbuffer = np.zeros(N)                                       # initialize buffer for FFT
  ysw = np.zeros(Ns)                                            # initialize output sound frame
  yrw = np.zeros(Ns)                                            # initialize output sound frame
  ys = np.zeros(x.size)                                         # initialize output array
  yr = np.zeros(x.size)                                         # initialize output array
  w = w / sum(w)                                                # normalize analysis window
  sw = np.zeros(Ns)     
  ow = triang(2*H)                                              # overlapping window
  sw[hNs-H:hNs+H] = ow      
  bh = blackmanharris(Ns)                                       # synthesis window
  bh = bh / sum(bh)                                             # normalize synthesis window
  wr = bh                                                       # window for residual
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]

  while pin<pend:  
  #-----analysis-----             
    xw = x[pin-hM1:pin+hM2] * w                                  # window the input sound
    fftbuffer = np.zeros(N)                                      # reset buffer
    fftbuffer[:hM1] = xw[hM2:]                                   # zero-phase window in fftbuffer
    fftbuffer[N-hM2:] = xw[:hM2]                           
    X = fft(fftbuffer)                                           # compute FFT
    mX = 20 * np.log10(abs(X[:hN]))                              # magnitude spectrum of positive frequencies
    ploc = PP.peakDetection(mX, hN, t)                
    pX = np.unwrap(np.angle(X[:hN]))                             # unwrapped phase spect. of positive freq.    
    iploc, ipmag, ipphase = PP.peakInterp(mX, pX, ploc)          # refine peak values
        
    iploc = (iploc!=0) * (iploc*Ns/N)                            # synth. locs
    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xr = x[ri:ri+Ns]*wr                                          # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xr[hNs:]                                   # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xr[:hNs]                           
    Xr = fft(fftbuffer)                                          # compute FFT for residual analysis
  
  #-----synthesis-----
    Ys = GS.genSpecSines(iploc, ipmag, ipphase, Ns)              # generate spec of sinusoidal component          
    Yr = Xr-Ys;                                                  # get the residual complex spectrum

    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Ys))                                # inverse FFT of sinusoidal spectrum
    ysw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    ysw[hNs-1:] = fftbuffer[:hNs+1] 
    
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Yr))                                # inverse FFT of residual spectrum
    yrw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    yrw[hNs-1:] = fftbuffer[:hNs+1]
    
    ys[ri:ri+Ns] += sw*ysw                                       # overlap-add for sines
    yr[ri:ri+Ns] += sw*yrw                                       # overlap-add for residual
    pin += H                                                     # advance sound pointer
  
  y = ys+yr                                                      # sum of sinusoidal and residual components
  return y, ys, yr


def defaultTest():
    str_time = time.time()
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/mridangam.wav'))
    w = np.blackman(901)
    N = 1024
    t = -70
    y, ys, yr = sprModel(x, fs, w, N, t)
    print "time taken for computation " + str(time.time()-str_time)
  
if __name__ == '__main__':
    
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/mridangam.wav'))
    w = np.blackman(601)
    N = 2048
    t = -70
    y, ys, yr = sprModel(x, fs, w, N, t)

    wp.play(y, fs)
    wp.play(ys, fs)
    wp.play(yr, fs)