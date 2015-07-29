import matplotlib.pyplot as plt
import numpy as np

A = .8
f0 = 1000
phi = np.pi/2
fs = 44100
t = np.arange(-.002, .002, 1.0/fs)
x = A * np.cos(2*np.pi*f0*t+phi)

plt.figure(1, figsize=(9.5, 3.5))
plt.plot(t, x, linewidth=2)
plt.axis([-.002,.002,-0.8,0.8])
plt.xlabel('time')
plt.ylabel('amplitude')

plt.tight_layout()
plt.savefig('sinewave.png')
plt.show()
