import numpy as np
import smsWavplayer as wp
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft, ifft
import time, os, sys

def stft(x, fs, w, N, H) :
  # Analysis/synthesis of a sound using the short-time fourier transform
  # x: input array sound, w: analysis window (odd size), N: FFT size, H: hop size
  # returns y: output array sound

  hN = N/2                                                # size of positive spectrum
  hM = (w.size+1)/2                                       # half analysis window size
  pin = hM                                                # initialize sound pointer in middle of analysis window       
  pend = x.size-hM                                        # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  yw = np.zeros(w.size)                                   # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window

  while pin<pend:                                         # while sound pointer is smaller than last sample      
            
  #-----analysis-----             
    xw = x[pin-hM:pin+hM-1] * w                           # window the input sound
    fftbuffer = np.zeros(N)                               # clean fft buffer
    fftbuffer[:hM] = xw[hM-1:]                            # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]        
    X = fft(fftbuffer)                                    # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                     # magnitude spectrum of positive frequencies in dB     
    pX = np.unwrap( np.angle(X[:hN]) )                    # unwrapped phase spectrum of positive frequencies
    
  #-----synthesis-----
    Y = np.zeros(N, dtype = complex)                      # clean output spectrun
    Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                  # generate positive frequencies
    Y[hN+1:] = 10**(mX[:0:-1]/20) * np.exp(-1j*pX[:0:-1]) # generate negative frequencies
    fftbuffer = np.real( ifft(Y) )                        # compute inverse FFT
    yw[:hM-1] = fftbuffer[N-hM+1:]                        # undo zero-phase window
    yw[hM-1:] = fftbuffer[:hM]
    y[pin-hM:pin+hM-1] += H*yw                            # overlap-add
    pin += H                                              # advance sound pointer
  
  return y

def defaultTest():
  str_time = time.time()    
  (fs, x) = wp.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../sounds/oboe.wav'))
  w = np.hamming(511)
  N = 512
  H = 256
  y = stft(x, fs, w, N, H)
  print "time taken for computation " + str(time.time()-str_time)  

# example call of stft function
if __name__ == '__main__':
  (fs, x) = wp.wavread('../../sounds/oboe.wav')
  w = np.hamming(511)
  N = 512
  H = 256
  y = stft(x, fs, w, N, H)
  wp.play(y, fs)