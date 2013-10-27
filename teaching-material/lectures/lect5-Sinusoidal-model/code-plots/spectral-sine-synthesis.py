import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
from scipy.fftpack import fft, ifft, fftshift

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

iploc = np.array([20.5, 130.3, 200.2])
ipmag = np.array([-2.2, -4.3, -8.2])
ipphase = np.array([1.1, 0.2, 2.4])

N = 512
hN = N/2

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



mY = 20 * np.log10( abs(Y) )                     # magnitude spectrum of positive frequencies
pY = np.unwrap( np.angle(Y) ) 

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(np.arange(-hN, hN), fftshift(mY))
plt.axis([-hN, hN,-100,0])
plt.title("mag spectrum: fr = 20.5, 130.3, 200.2; Ar = -2.2, -4.3, -8.2")

plt.subplot(3,1,2)
plt.plot(np.arange(-hN, hN), fftshift(pY))
plt.axis([-hN, hN,-3.14,3.14])
plt.title("phase spectrum: pr= 1.1, 0.2, 2.4")

plt.subplot(3,1,3)
plt.plot(np.arange(-hN, hN), 32*fftshift(ifft(Y)))
plt.axis([-hN, hN,-1,1])
plt.title("synthesize sound")

plt.show()