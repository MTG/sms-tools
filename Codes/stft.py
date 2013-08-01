import numpy as np
import matplotlib.pyplot as plt
import wavplayer as wp
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft, ifft
import time

def stft(x, fs, w, N, H) :
  # Analysis/synthesis of a sound using the short-time fourier transform
  # x: input array sound, w: analysis window, N: FFT size, H: hop size
  # y: output sound

  hN = N/2                                                # size of positive spectrum
  hM = (w.size+1)/2                                       # half analysis window size
  pin = hM                                                # initialize sound pointer in middle of analysis window       
  pend = x.size-hM                                        # last sample to start a frame
  fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
  x = np.float32(x) / (2**15)                             # normalize input signal (so 0 dB is the max amp. value)
  yw = np.zeros(w.size)                                   # initialize output sound frame
  y = np.zeros(x.size)                                    # initialize output array
  w = w / sum(w)                                          # normalize analysis window

  while pin<pend:       
            
  #-----analysis-----             
    xw = x[pin-hM:pin+hM-1] * w                           # window the input sound
    fftbuffer = np.zeros(N)                               # reset buffer
    fftbuffer[:hM] = xw[hM-1:]                            # zero-phase window in fftbuffer
    fftbuffer[N-hM+1:] = xw[:hM-1]        
    X = fft(fftbuffer)                                    # compute FFT
    mX = 20 * np.log10( abs(X[:hN]) )                     # magnitude spectrum of positive frequencies
    # fig.clf()       
    # plt.plot(xw, color = 'r')       
    pX = np.unwrap( np.angle(X[:hN]) )                    # unwrapped phase spect. of positive freq.
    
  #-----synthesis-----
    Y = np.zeros(N, dtype = complex)
    Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                # generate positive freq.
    Y[hN+1:] = 10**(mX[:0:-1]/20) * np.exp(-1j*pX[:0:-1])   # generate neg.freq.
    fftbuffer = np.real( ifft(Y) )                        # inverse FFT
    yw[:hM-1] = fftbuffer[N-hM+1:]                            # undo zero-phase window
    yw[hM-1:] = fftbuffer[:hM] 
    # plt.plot(yw, color = 'b')
    # plt.draw()
    # time.sleep(0.01)
    y[pin-hM:pin+hM-1] += H*yw                           # overlap-add
    pin += H                                              # advance sound pointer
  
  return y

(fs, x) = read('oboe.wav')
w = np.hamming(511)
N = 512
H = 256
wp.play(x, fs, w, N, H)
# fig = plt.figure()

y = stft(x)
y *= 2**15
y = y.astype(np.int16)
wp.play(y, fs)