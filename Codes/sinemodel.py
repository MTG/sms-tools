import numpy as np
import UtilityFunctions as uf
import matplotlib.pyplot as plt
import wavplayer as wp
from scipy.io.wavfile import read
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft

def genspecsines(iploc, ipmag, ipphase, N):
  # Generate a spectrum from a series of sine values
  # iploc, ipmag, ipphase: sine peaks locations, magnitudes and phases
  # N: size of complex spectrum
  # returns Y: generated complex spectrum of sines

  Y = np.zeros(N, dtype = complex)                 # initialize output spectrum  
  hN = N/2                                         # size of positive freq. spectrum

  for i in range(0, iploc.size):                   # generate all sine spectral lobes
    loc = iploc[i]                                 # it should be in range ]0,hN-1[

    if loc<1 or loc>hN-1: continue
    binremainder = round(loc)-loc;
    lb = np.arange(binremainder-4, binremainder+5) # main lobe (real value) bins to read
    lmag = uf.genbh92lobe(lb) * 10**(ipmag[i]/20)  # lobe magnitudes of the complex exponential
    b = np.arange(round(loc)-4, round(loc)+5)
    
    for m in range(0, 9):
      if b[m] < 0:                                 # peak lobe crosses DC bin
        Y[-b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
      elif b[m] > hN:                              # peak lobe croses Nyquist bin
        Y[b[m]] += lmag[m]*np.exp(-1j*ipphase[i])
      elif b[m] == 0 or b[m] == hN:                # peak lobe in the limits of the spectrum 
        Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i]) + lmag[m]*np.exp(-1j*ipphase[i])
      else:                                        # peak lobe in positive freq. range
        Y[b[m]] += lmag[m]*np.exp(1j*ipphase[i])
    
    Y[hN+1:] = Y[hN-1:0:-1].conjugate()            # fill the negative part of the spectrum
  
  return Y

def peak_interp(mX, pX, ploc):
  # interpolate peak values using parabolic interpolation
  # mX: magnitude spectrum, pX: phase spectrum, ploc: locations of peaks
  # returns iploc, ipmag, ipphase: interpolated values
  
  val = mX[ploc]                                          # magnitude of peak bin 
  lval = mX[ploc-1]                                       # magnitude of bin at left
  rval = mX[ploc+1]                                       # magnitude of bin at right
  iploc = ploc + 0.5*(lval-rval)/(lval-2*val+rval)        # center of parabola
  ipmag = val - 0.25*(lval-rval)*(iploc-ploc)             # magnitude of peaks
  ipphase = np.interp(iploc, np.arange(0, pX.size), pX)   # phase of peaks

  return iploc, ipmag, ipphase

def peak_detection(mX, hN, t):
  # detect spectral peak locations
  # mX: magnitude spectrum, hN: half number of samples, t: threshold
  # returns ploc: peak locations

  thresh = np.where(mX[1:hN-1]>t, mX[1:hN-1], 0);          # locations above threshold
  next_minor = np.where(mX[1:hN-1]>mX[2:], mX[1:hN-1], 0)  # locations higher than the next one
  prev_minor = np.where(mX[1:hN-1]>mX[:hN-2], mX[1:hN-1], 0) # locations higher than the previous one
  ploc = thresh * next_minor * prev_minor                  # locations fulfilling the three criteria
  ploc = ploc.nonzero()[0] + 1

  return ploc

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
    ploc = peak_detection(mX, hN, t)                      # detect locations of peaks
    pmag = mX[ploc]                                       # get the magnitude of the peajs
    pX = np.unwrap( np.angle(X[:hN]) )                    # unwrapped phase spect. of positive freq.
    iploc, ipmag, ipphase = peak_interp(mX, pX, ploc)     # refine peak values by interpolation
  
  #-----synthesis-----
    plocs = iploc*Ns/N;                                   # adapt peak locations to size of synthesis FFT
    Y = genspecsines(plocs, ipmag, ipphase, Ns)           # generate sines in the spectrum         
    fftbuffer = np.real( ifft(Y) )                        # compute inverse FFT
    yw[:hNs-1] = fftbuffer[hNs+1:]                        # undo zero-phase window
    yw[hNs-1:] = fftbuffer[:hNs+1] 
    y[pin-hNs:pin+hNs] += sw*yw                           # overlap-add and apply a synthesis window
    pin += H                                              # advance sound pointer
    
  return y

# example call of sine_model function
(fs, x) = read('../sounds/oboe.wav')
w = np.hamming(511)
N = 512
t = -60
fig = plt.figure()
y = sine_model(x, fs, w, N, t)

y *= 2**15
y = y.astype(np.int16)
wp.play(y, fs)