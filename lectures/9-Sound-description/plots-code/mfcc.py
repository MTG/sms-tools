import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess

M = 1024
N = 1024
H = 512
fs = 44100
spectrum = ess.Spectrum(size=N)
window = ess.Windowing(size=M, type='hann')
mfcc = ess.MFCC(numberCoefficients = 12)
x = ess.MonoLoader(filename = '../../../sounds/speech-male.wav', sampleRate = fs)()
mfccs = []

for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):          
  mX = spectrum(window(frame))
  mfcc_bands, mfcc_coeffs = mfcc(mX)
  mfccs.append(mfcc_coeffs)            
mfccs = np.array(mfccs)

plt.figure(1, figsize=(9.5, 7))

plt.subplot(2,1,1)
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.title('x (speech-male.wav)')

plt.subplot(2,1,2)
numFrames = int(mfccs[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                                                    
plt.pcolormesh(frmTime, 1+np.arange(12), np.transpose(mfccs[:,1:]))
plt.ylabel('coefficients')
plt.title('MFCCs')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('mfcc.png')
plt.show()

