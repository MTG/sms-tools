import matplotlib.pyplot as plt
import numpy as np

Ar = .8
fr = 2.0
phi = np.pi/2
fs = 100
t = np.arange(-1, 1, 1.0/fs)
x = Ar * np.cos(2*np.pi*fr*t+phi)
plt.plot(t, x, 'b')
plt.axis([-1, 1,-.8,.8])
plt.xlabel('time (seconds)')
plt.ylabel('amplitude')
plt.show()
