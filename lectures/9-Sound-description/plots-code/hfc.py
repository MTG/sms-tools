import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess

M = 1024
N = 1024
H = 512
fs = 44100
spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
hfc = ess.HFC()
x = ess.MonoLoader(filename = '../../../sounds/speech-male.wav', sampleRate = fs)()
hfcs = []

for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):  
  mX = spectrum(window(frame))        
  hfc_val = hfc(mX)
  hfcs.append(hfc_val)            
hfcs = np.array(hfcs)

plt.figure(1, figsize=(9.5, 5))
plt.subplot(2,1,1)

plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.title('x (speech-male.wav)')

plt.subplot(2,1,2)
frmTime = H*np.arange(hfcs.size)/float(fs)    
plt.plot(frmTime, hfcs, 'g', lw=1.5)  
plt.axis([0, x.size/float(fs), min(hfcs), max(hfcs)])
plt.xlabel('time (sec)')
plt.ylabel('high frequency content')

plt.tight_layout()
plt.savefig('hfc.png')
plt.show()
