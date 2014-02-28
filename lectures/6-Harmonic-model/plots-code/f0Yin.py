import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
import sys, os
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions_C/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import waveIO as WIO
import stftAnal as ST


def f0Yin(x, N, H, minf0, maxf0):
  # fundamental frequency detection using the Yin algorithm
  # x: input sound, N: window size,
  # minf0: minimum f0 frequency in Hz, maxf0: maximim f0 frequency in Hz, 
  # returns f0

  spectrum = ess.Spectrum(size=N)
  window = ess.Windowing(size=N, type='hann')
  pitchYin= ess.PitchYinFFT(minFrequency = minf0, maxFrequency = maxf0)
  pin = 0
  pend = x.size-N
  f0 = []

  while pin<pend:             
    mX = spectrum(window(x[pin:pin+N]))
    f0t = pitchYin(mX)
    f0 = np.append(f0, f0t[0])
    pin += H                     
  return f0

if __name__ == '__main__':
  (fs, x) = WIO.wavread('../../../sounds/vignesh.wav')

  N = 2048
  H = 256
  w = hamming(2048)
  mX, pX = ST.stftAnal(x, fs, w, N, H)
  maxplotfreq = 2000.0
  frmTime = H*np.arange(mX[:,0].size)/float(fs)                             
  binFreq = fs*np.arange(N*maxplotfreq/fs)/N                        
  plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
  
  N = 2048
  minf0 = 130
  maxf0 = 300
  H = 256
  f0 = f0Yin(x, N, H, minf0, maxf0)
  frmTime = H*np.arange(f0.size)/float(fs)  
  plt.plot(frmTime, f0, linewidth=2, color='k')

  plt.autoscale(tight=True)
  plt.title('f0 on spectrogram; YIN: N=2048, H = 256')
  plt.show()

