import sys, csv, os
from essentia import *
from essentia.standard import *
from pylab import *
from numpy import *

filename = '../../../sounds/vignesh.wav'
fs = 44100
H = 128
M = 2048


predominantMelody = PredominantMelody(frameSize=M, hopSize=H)
x = MonoLoader(filename = filename, sampleRate = fs)()

pitch, pitchConfidence = predominantMelody(x)


plt.figure(1, figsize=(9.5, 4))
plt.subplot(2,1,1)

plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.title('x (carnatic.wav)')

plt.subplot(2,1,2)
frmTime = H*np.arange(pitch.size)/float(fs) 
pitch[pitch==0]=nan
plot(frmTime, pitch, color='g', linewidth = 1.5)
plt.axis([0, x.size/float(fs), 100, 300])
plt.title('prominent melody')

tight_layout()
savefig('predominantMelody.png')

show()
