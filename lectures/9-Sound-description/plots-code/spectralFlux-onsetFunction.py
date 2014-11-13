import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess

M = 1024
N = 1024
H = 512
fs = 44100
spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
flux = ess.Flux()
onsetDetection = ess.OnsetDetection(method='hfc')
x = ess.MonoLoader(filename = '../../../sounds/speech-male.wav', sampleRate = fs)()
fluxes = []
onsetDetections = []

for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):  
  mX = spectrum(window(frame))        
  flux_val = flux(mX)
  fluxes.append(flux_val)
  onsetDetection_val = onsetDetection(mX, mX)
  onsetDetections.append(onsetDetection_val)            
onsetDetections = np.array(onsetDetections)            
fluxes = np.array(fluxes)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(2,1,1)

plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.title('x (speech-male.wav)')

plt.subplot(2,1,2)
frmTime = H*np.arange(fluxes.size)/float(fs)    
plt.plot(frmTime, fluxes/max(fluxes), 'g', lw=1.5, label ='normalized spectral flux')  
plt.plot(frmTime, onsetDetections/max(onsetDetections), 'c', lw=1.5, label = 'normalized onset detection')  
plt.axis([0, x.size/float(fs), 0, 1])
plt.legend()

plt.tight_layout()
plt.savefig('spectralFlux-onsetFunction.png')
plt.show()
