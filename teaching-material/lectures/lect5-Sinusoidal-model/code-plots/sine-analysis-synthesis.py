import numpy as np
import matplotlib.pyplot as plt
import wavplayer as wp
from scipy.io.wavfile import read
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft

def genspecsines(iploc, ipmag, ipphase, N):
  # Compute a spectrum from a series of sine values
  # iploc, ipmag, ipphase: sine locations, magnitudes and phases
  # N: size of complex spectrum
  # Y: generated complex spectrum of sines

  Y = np.zeros(N, dtype = complex)                 # initialize output spectrum  
  hN = N/2                                         # size of positive freq. spectrum

  for i in range(0, iploc.size):                   # generate all sine spectral lobes
    loc = iploc[i]                                 # it should be in range ]0,hN-1[

    if loc<1 or loc>hN-1: continue
    binremainder = round(loc)-loc;
    lb = np.arange(binremainder-4, binremainder+5) # main lobe (real value) bins to read
    lmag = genbh92lobe(lb) * 10**(ipmag[i]/20)     # lobe magnitudes of the complex exponential
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
    
    Y[hN+1:] = Y[hN-1:0:-1].conjugate()            # fill the rest of the spectrum
  
  return Y

def genbh92lobe(x):
  # Calculate transform of the Blackman-Harris 92dB window
  # x: bin positions to compute (real values), y: transform values

  N = 512;
  f = x*np.pi*2/N                                  # frequency sampling
  df = 2*np.pi/N  
  y = np.zeros(x.size)                               # initialize window
  consts = [0.35875, 0.48829, 0.14128, 0.01168]      # window constants
  
  for m in range(0,4):  
    y += consts[m]/2 * (D(f-df*m, N) + D(f+df*m, N)) # sum Dirichlet kernels
  
  y = y/N/consts[0] 
  
  return y                                           # normalize

def D(x, N):
  # Calculate rectangular window transform (Dirichlet kernel)

  y = np.sin(N * x/2) / np.sin(x/2)
  y[np.isnan(y)] = N                                 # avoid NaN if x == 0
  
  return y

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


M = 801
w = np.hamming(M)
N = 1024
t = -40
freqs = np.array([100.2, 200.3, 300.2])
amps = np.array([.99, .7, .5]) * 2**15
phases = np.array([1.1, 3.2, 2.4])
fs = 2000
Ns = 512

x = amps[0] * np.cos(2*np.pi*freqs[0]/fs*np.arange(M) + phases[0]) + amps[1] * np.cos(2*np.pi*freqs[1]/fs*np.arange(M) + phases[1]) + amps[2] * np.cos(2*np.pi*freqs[2]/fs*np.arange(M) + phases[2])

hN = N/2                                                # size of positive spectrum
hM = (w.size+1)/2                                       # half analysis window size                                                # FFT size for synthesis (even)
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

xw = x * w                                            # window the input sound
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
y = sw*yw                           # overlap-add and apply a synthesis window

plt.figure(1)

plt.subplot(4,1,1)
plt.plot(x/max(x))
plt.axis([0, M,-1,1])
plt.title("input sound")

plt.subplot(4,1,2)
plt.plot(mX)
plt.plot(iploc, ipmag, 'ro') 
plt.axis([0, hN,-70,0])
plt.title("analysis spectrum")

mY = 20 * np.log10( abs(Y) ) 
plt.subplot(4,1,3)
plt.plot(mY[0:hNs])
plt.axis([0, hNs,-70,0])
plt.title("synthesys spectrum")

plt.subplot(4,1,4)
plt.plot(y/max(y))
plt.axis([0, Ns,-1,1])
plt.title("output sound")

plt.show()