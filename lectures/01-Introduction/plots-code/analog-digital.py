import matplotlib.pyplot as plt
import numpy as np

plt.figure(1, figsize=(9.5, 6))

ax1=plt.subplot(2, 1, 1)
A = .8
f0 = 2
phi = np.pi/2
fs = 100
t = np.arange(-1, 1, 1.0/fs)
x = A * np.cos(2*np.pi*f0*t+phi)
ax1.plot(t, x, 'b',lw=2)
plt.axis([-1,1,-0.8,0.8])
ax1.set_title('analog')
ax1.set_xlabel('time (sec)')
ax1.set_ylabel('amplitude')
ax2=plt.subplot(212, sharex=ax1)
ax2.plot(t, x, '*', lw=2)
plt.axis([-1,1,-0.8,0.8])
plt.grid(True)
plt.title('digital')
plt.xlabel('time (sec)')
plt.ylabel('amplitude')

plt.tight_layout()
plt.savefig('analog-digital.png')
plt.show()
