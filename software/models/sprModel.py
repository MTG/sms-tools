# functions that implement analysis and synthesis of sounds using the Sinusoidal plus Residual Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal import blackmanharris, triang
from scipy.fftpack import fft, ifft, fftshift
import math
import sineModel as SM
import stft as STFT
import utilFunctions as UF
  
  
def sprModel(x, fs, w, N, t):
  # Analysis/synthesis of a sound using the sinusoidal plus residual model, one frame at a time
  # x: input sound, fs: sampling rate, w: analysis window, 
  # N: FFT size (minimum 512), t: threshold in negative dB, 
  # returns y: output sound, ys: sinusoidal component, xr: residual component

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
  xrw = np.zeros(Ns)                                            # initialize output sound frame
  ys = np.zeros(x.size)                                         # initialize output array
  xr = np.zeros(x.size)                                         # initialize output array
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
    ploc = UF.peakDetection(mX, hN, t)                
    pX = np.unwrap(np.angle(X[:hN]))                             # unwrapped phase spect. of positive freq.    
    iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)          # refine peak values
    iploc = (iploc!=0) * (iploc*Ns/N)                            # synth. locs
    ri = pin-hNs-1                                               # input sound pointer for residual analysis
    xw2 = x[ri:ri+Ns]*wr                                          # window the input sound                                       
    fftbuffer = np.zeros(Ns)                                     # reset buffer
    fftbuffer[:hNs] = xw2[hNs:]                                   # zero-phase window in fftbuffer
    fftbuffer[hNs:] = xw2[:hNs]                           
    X2 = fft(fftbuffer)                                          # compute FFT for residual analysis
  #-----synthesis-----
    Ys = UF.genSpecSines(fs*iploc/N, ipmag, ipphase, Ns, fs)     # generate spec of sinusoidal component          
    Xr = X2-Ys;                                                  # get the residual complex spectrum
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Ys))                                # inverse FFT of sinusoidal spectrum
    ysw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    ysw[hNs-1:] = fftbuffer[:hNs+1] 
    fftbuffer = np.zeros(Ns)
    fftbuffer = np.real(ifft(Xr))                                # inverse FFT of residual spectrum
    xrw[:hNs-1] = fftbuffer[hNs+1:]                              # undo zero-phase window
    xrw[hNs-1:] = fftbuffer[:hNs+1]
    ys[ri:ri+Ns] += sw*ysw                                       # overlap-add for sines
    xr[ri:ri+Ns] += sw*xrw                                       # overlap-add for residual
    pin += H                                                     # advance sound pointer
  y = ys+xr                                                      # sum of sinusoidal and residual components
  return y, ys, xr
