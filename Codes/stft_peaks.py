import numpy as np
import matplotlib.pyplot as plt
import wavplayer as wp
from scipy.io.wavfile import read
from scipy.signal import hamming
from scipy.fftpack import fft, ifft
import time

def peak_detection(mX, hN, t) :
  # mX: magnitude spectrum, hN: half number of samples, t: threshold in negative dB
  # to be a peak it has to accomplish three conditions:

  thresh = np.where(mX[1:hN-1]>t, mX[1:hN-1], 0);             # more than threshold
  next_minor = np.where(mX[1:hN-1]>mX[2:], mX[1:hN-1], 0)     # next value less than actual
  prev_minor = np.where(mX[1:hN-1]>mX[:hN-2], mX[1:hN-1], 0)  # previous value less than actual
  ploc = thresh * next_minor * prev_minor
  ploc = ploc.nonzero()[0] + 1

  return ploc

def stft_peaks(x, fs, w, N, H, t) :
  # Analysis/synthesis of a sound using the peaks
  # x: input array sound, w: analysis window, N: FFT size, H: hop size, 
  # t: threshold in negative dB, y: output sound

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
    ploc = peak_detection(mX, hN, t)
    pmag = mX[ploc]
    # freq = np.arange(0, fs/2, fs/N)                     # frequency axis in Hz
    # freq = freq[:freq.size-1]
    # fig.clf()
    # plt.plot(freq, mX)
    # plt.ylabel('Magnitude(dB)'), plt.xlabel('Frequency(Hz)')         
    # plt.plot(freq[ploc], pmag, 'ro')
    # plt.draw()          
    pX = np.unwrap( np.angle(X[:hN]) )                    # unwrapped phase spect. of positive freq.
    pphase = pX[ploc]

  #-----synthesis-----
    Y = np.zeros(N, dtype = complex)
    Y[ploc] = 10**(pmag/20) * np.exp(1j*pphase)           # generate positive freq.
    Y[N-ploc] = 10**(pmag/20) * np.exp(-1j*pphase)        # generate neg.freq.
    fftbuffer = np.real( ifft(Y) )                        # inverse FFT
    yw[:hM-1] = fftbuffer[N-hM+1:]                        # undo zero-phase window
    yw[hM-1:] = fftbuffer[:hM] 
    y[pin-hM:pin+hM-1] += H*yw                            # overlap-add
    pin += H                                              # advance sound pointer
  
  return y
  
(fs, x) = read('oboe.wav')
wp.play(x, fs)
w = np.hamming(511)
N = 512
H = 256
t = -60
# fig = plt.figure()
y = stft_peaks(x, fs, w, N, H, t)

y *= 2**15
y = y.astype(np.int16)
wp.play(y, fs)