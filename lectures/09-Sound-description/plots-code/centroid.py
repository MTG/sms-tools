import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess

M = 1024
N = 1024
H = 512
fs = 44100
spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
centroid = ess.Centroid(range=fs/2.0)
x = ess.MonoLoader(filename = '../../../sounds/speech-male.wav', sampleRate = fs)()
centroids = []

for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):  
  mX = spectrum(window(frame))        
  centroid_val = centroid(mX)
  centroids.append(centroid_val)            
centroids = np.array(centroids)

plt.figure(1, figsize=(9.5, 5))
plt.subplot(2,1,1)

plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.title('x (speech-male.wav)')

plt.subplot(2,1,2)
frmTime = H*np.arange(centroids.size)/float(fs)    
plt.plot(frmTime, centroids, 'g', lw=1.5)  
plt.axis([0, x.size/float(fs), min(centroids), max(centroids)])
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectral centroid')

plt.tight_layout()
plt.savefig('centroid.png')
plt.show()
