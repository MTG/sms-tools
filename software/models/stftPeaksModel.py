import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))

import dftAnal
import waveIO as wp
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft, ifft
import math
import peakProcessing as PP 

def stftPeaksModel(x, fs, w, N, H, t) :
  # Analysis/synthesis of a sound using the spectral peaks
  # x: input array sound, w: analysis window, N: FFT size, H: hop size, 
  # t: threshold in negative dB 
  # returns y: output array sound

  hN = N/2    
  hM1 = int(math.floor((w.size+1)/2))                     # half analysis window size by rounding
  hM2 = int(math.floor(w.size/2))                         # half analysis window size by floor
  pin = hM1                                               # initialize sound pointer in middle of analysis window       
  pend = x.size-hM1                                       # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yw = np.zeros(w.size)                                   # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window
  
  while pin<pend:       
           
  #-----analysis-----             
    x1 = x[pin-hM1:pin+hM2]                               # select frame
    mX, pX = dftAnal.dftAnal(x1, w, N)                    # compute dft
    ploc = PP.peakDetection(mX, hN, t)                    # detect all peaks above a threshold
    pmag = mX[ploc]                                       # get the magnitude of the peaks
    pphase = pX[ploc]

  #-----synthesis-----
    Y = np.zeros(N, dtype = complex)
    Y[ploc] = 10**(pmag/20) * np.exp(1j*pphase)           # generate positive freq.
    Y[N-ploc] = 10**(pmag/20) * np.exp(-1j*pphase)        # generate neg.freq.
    fftbuffer = np.real(ifft(Y))                          # inverse FFT
    yw[:hM2] = fftbuffer[N-hM2:]                          # undo zero-phase window
    yw[hM2:] = fftbuffer[:hM1]
    y[pin-hM1:pin+hM2] += H*yw                            # overlap-add
    pin += H                                              # advance sound pointer
  
  return y


def defaultTest():
    str_time = time.time()
    (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/oboe-A4.wav'))
    w = np.hamming(801)
    N = 1024
    H = 200
    t = -70
    y = stftPeaksModel(x, fs, w, N, H, t)
    print "time taken for computation " + str(time.time()-str_time)
    
  
if __name__ == '__main__':   
      
    (fs, x) = wp.wavread('../../sounds/oboe-A4.wav')
    w = np.hamming(801)
    N = 1024
    H = 200
    t = -70
    y = stftPeaksModel(x, fs, w, N, H, t)
    wp.play(y, fs)