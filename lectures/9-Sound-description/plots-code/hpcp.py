import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess

M = 1024
N = 1024
H = 512
fs = 44100
spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
spectralPeaks = ess.SpectralPeaks()
hpcp = ess.HPCP()
x = ess.MonoLoader(filename = '../../../sounds/cello-double.wav', sampleRate = fs)()
hpcps = []

for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):          
  mX = spectrum(window(frame))
  spectralPeaks_freqs, spectralPeaks_mags = spectralPeaks(mX) 
  hpcp_vals = hpcp(spectralPeaks_freqs, spectralPeaks_mags)
  hpcps.append(hpcp_vals)            
hpcps = np.array(hpcps)

plt.figure(1, figsize=(9.5, 7))

plt.subplot(2,1,1)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.title('x (cello-double.wav)')

plt.subplot(2,1,2)
numFrames = int(hpcps[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                                                    
plt.pcolormesh(frmTime, np.arange(12), np.transpose(hpcps))
plt.ylabel('spectral bins')
plt.title('HPCP')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('hpcp.png')
plt.show()

