import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import ifft

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
    lmag = uf.genbh92lobe(lb) * 10**(ipmag[i]/20)     # lobe magnitudes of the complex exponential
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

Ns = 256
fs = 10000.0
freqs = np.array([500])
mags = np.array([.8])
phases = np.array([1.4])

Y = genbh92lobe(np.arange(0,Ns))
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(np.arange(0, fs/2.0, fs/(float(Ns))), np.abs(Y), 'ro')
plt.axis([0,fs/2.0,0,Ns])
plt.xlabel('frenquency (Hz)')
plt.ylabel('amplitude')

plt.subplot(3,1,2)
plt.plot(np.arange(0, fs/2.0, fs/(float(Ns))), no.angle(Y), 'bo')
plt.axis([0,fs/2.0,0,2*np.pi])
plt.xlabel('frenquency (Hz)')
plt.ylabel('phase (radians)')

y = np.real(ifft(Y))     
yw[:hNs-1] = y[hNs+1:] 
yw[hNs-1:] = y[:hNs+1] 

plt.subplot(3,1,3)
plt.plot(np.arange(-hNs/float(fs), hNs/float(fs), 1.0/(fs)), yw, 'r')
plt.axis([-hNs/float(fs), (hNs-1)/float(fs),-1,1])
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')