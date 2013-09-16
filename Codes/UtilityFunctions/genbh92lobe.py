import numpy as np


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