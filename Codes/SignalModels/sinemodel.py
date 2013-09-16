import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft


import sys, os, functools, time

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
  

def sine_model(x, fs, w, N, t):
  # Analysis/synthesis of a sound using the sinusoidal model
  # x: input array sound, w: analysis window, N: size of complex spectrum,
  # t: threshold in negative dB 
  # returns y: output array sound

  hN = N/2                                                # size of positive spectrum
  hM = (w.size+1)/2                                       # half analysis window size
  Ns = 512                                                # FFT size for synthesis (even)
  H = Ns/4                                                # Hop size used for analysis and synthesis
  hNs = Ns/2                                              # half of synthesis FFT size
  pin = max(hNs, hM)                                      # initialize sound pointer in middle of analysis window       
  pend = x.size - max(hNs, hM)                            # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yw = np.zeros(Ns)                                       # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  x = np.float32(x) / (2**15)                             # normalize input signal
  w = w / sum(w)                                          # normalize analysis window
  sw = np.zeros(Ns)                                       # initialize synthesis window
  ow = triang(2*H);                                       # triangular window
  sw[hNs-H:hNs+H] = ow                                    # add triangular window
  bh = blackmanharris(Ns)                                 # blackmanharris window
  bh = bh / sum(bh)                                       # normalized blackmanharris window
  sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]     # normalized synthesis window

  while pin<pend:                                         # while input sound pointer is within sound 
    
  #-----analysis-----             
    xw = x[pin-hM:pin+hM-1] * w                           # window the input sound
    fftbuffer = np.zeros(N)                               # reset buffer
    fftbuffer[:hM] = xw[hM-1:]                            # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]        
    X = fft(fftbuffer)                                    # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                     # magnitude spectrum of positive frequencies
    ploc = PP.peak_detection(mX, hN, t)                      # detect locations of peaks
    pmag = mX[ploc]                                       # get the magnitude of the peajs
    pX = np.unwrap( np.angle(X[:hN]) )                    # unwrapped phase spect. of positive freq.
    iploc, ipmag, ipphase = PP.peak_interp(mX, pX, ploc)     # refine peak values by interpolation
  
  #-----synthesis-----
    plocs = iploc*Ns/N;                                   # adapt peak locations to size of synthesis FFT
    Y = GS.genspecsines(plocs, ipmag, ipphase, Ns)           # generate sines in the spectrum         
    fftbuffer = np.real( ifft(Y) )                        # compute inverse FFT
    yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yw[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
    pin += H                                              # advance sound pointer
    
  return y

def DefaultTest():
  
  str_time = time.time()
    
  (fs, x) = read('../../sounds/oboe.wav')
  w = np.hamming(511)
  N = 512
  t = -60
  fig = plt.figure()
  y = sine_model(x, fs, w, N, t)

  y *= 2**15
  y = y.astype(np.int16)
  
  print "time taken for computation " + str(time.time()-str_time)  
  
# example call of sine_model function
if __name__ == '__main__':
  (fs, x) = read('../../sounds/oboe.wav')
  w = np.hamming(511)
  N = 512
  t = -60
  fig = plt.figure()
  y = sine_model(x, fs, w, N, t)

  y *= 2**15
  y = y.astype(np.int16)
  wp.play(y, fs)