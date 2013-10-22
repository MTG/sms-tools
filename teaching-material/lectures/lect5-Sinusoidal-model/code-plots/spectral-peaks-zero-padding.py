import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.io.wavfile import read

def peak_interp(mX, pX, ploc):
  # mX: magnitude spectrum, pX: phase spectrum, ploc: locations of peaks
  # iploc, ipmag, ipphase: interpolated values
  
  val = mX[ploc]                                          # magnitude of peak bin 
  lval = mX[ploc-1]                                       # magnitude of bin at left
  rval = mX[ploc+1]                                       # magnitude of bin at right
  iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
  ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
  ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks

  return iploc, ipmag, ipphase

def peak_detection(mX, hN, t):
  # mX: magnitude spectrum, hN: half number of samples, t: threshold
  # to be a peak it has to accomplish three conditions: 

  thresh = np.where(mX[1:hN-1]>t, mX[1:hN-1], 0);
  next_minor = np.where(mX[1:hN-1]>mX[2:], mX[1:hN-1], 0)
  prev_minor = np.where(mX[1:hN-1]>mX[:hN-2], mX[1:hN-1], 0)
  ploc = thresh * next_minor * prev_minor
  ploc = ploc.nonzero()[0] + 1

  return ploc

plt.figure(1)

(fs, x) = read('oboe.wav')
N = 512
M = 511
t = 30
w = np.hamming(M)
start = .8*fs
hN = N/2                                                # size of positive spectrum
hM = (M+1)/2
xw = x[start:start+M] * w                           # window the input sound
fftbuffer = np.zeros(N)                               # reset buffer
fftbuffer[:hM] = xw[hM-1:]                            # zero-phase window in fftbuffer
fftbuffer[N-hM+1:] = xw[:hM-1]        
X = fft(fftbuffer)                                    # compute FFT
mX = 20 * np.log10( abs(X[:hN]/M) )                     # magnitude spectrum of positive frequencies
ploc = peak_detection(mX, hN, t)
pmag = mX[ploc]
freq = np.arange(0, fs/2, fs/N)                     # frequency axis in Hz
freq = freq[:freq.size-1] 
plt.subplot (2,1,1)
plt.plot(freq, mX)
plt.axis([300,2400,0,max(mX)+2])
plt.title('Spectral peaks: M=511, N=512')
plt.plot(freq[ploc], pmag, 'ro')    

N = 1024
M = 511
t = 30
w = np.hamming(M)
start = .8*fs
hN = N/2                                                # size of positive spectrum
hM = (M+1)/2
xw = x[start:start+M] * w                           # window the input sound
fftbuffer = np.zeros(N)                               # reset buffer
fftbuffer[:hM] = xw[hM-1:]                            # zero-phase window in fftbuffer
fftbuffer[N-hM+1:] = xw[:hM-1]        
X = fft(fftbuffer)                                    # compute FFT
mX = 20 * np.log10( abs(X[:hN]/M) )                     # magnitude spectrum of positive frequencies
ploc = peak_detection(mX, hN, t)
pmag = mX[ploc]
freq = np.arange(0, fs/2, fs/N)                     # frequency axis in Hz
freq = freq[:freq.size-1]
plt.subplot (2,1,2)
plt.plot(freq, mX)
plt.axis([300,2400,0,max(mX)+2])
plt.title('Spectral peaks: M=511, N=1024')
plt.plot(freq[ploc], pmag, 'ro')    


 
plt.show()