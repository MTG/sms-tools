import numpy as np
import matplotlib.pyplot as plt
import essentia.standard as ess

M = 1024
N = 1024
H = 512
fs = 44100
energy = ess.Energy()
rms = ess.RMS()
loudness = ess.Loudness()

x = ess.MonoLoader(filename = '../../../sounds/piano.wav', sampleRate = fs)()
energies = []
rmss = []
loudnesses = []

for frame in ess.FrameGenerator(x, frameSize=M, hopSize=H, startFromZero=True):          
  energy_val = energy(frame)
  energies.append(energy_val)
  rms_val = rms(frame)
  rmss.append(rms_val)
  loudness_val = loudness(frame)
  loudnesses.append(loudness_val)
          
energies = np.array(energies)/max(energies)
rmss = np.array(rmss)/max(rmss)
loudnesses = np.array(loudnesses)/max(loudnesses)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(2,1,1)

plt.plot(np.arange(x.size)/float(fs), x)
plt.axis([0, x.size/float(fs), min(x), max(x)])
plt.ylabel('amplitude')
plt.title('x (piano.wav)')

plt.subplot(2,1,2)
frmTime = H*np.arange(energies.size)/float(fs)    
plt.plot(frmTime, rmss, 'g', lw=1.5, label='normalized RMS')  
plt.plot(frmTime, loudnesses, 'c', lw=1.5, label ='normalized loudness')                       
plt.plot(frmTime, energies, 'r', lw=1.5, label='normalized energy')
plt.axis([0, x.size/float(fs), 0, 1])
plt.ylabel('normalized value')
plt.legend()

plt.tight_layout()
plt.savefig('loudness.png')
plt.show()

