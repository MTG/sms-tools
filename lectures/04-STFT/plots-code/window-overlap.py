import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


plt.figure(1, figsize=(9.5, 6))

N= 1000
M = 201
w = signal.blackman(M)
w1 = w/sum(w)

y = np.zeros(N)
H = 100
pin = 0
pend = N - M
plt.subplot(211)
while pin<pend:
	y [pin:pin+M] += w1*H
	plt.plot(np.arange(pin, pin+M), w, 'b', lw=1.5)
	pin += H
plt.plot(np.arange(0, N), y, 'r', lw=1.5)
plt.axis([0, N-H, 0, max(y)+.01])
plt.title('Blackman, M=201, H=100')

y = np.zeros(N)
H = 50
pin = 0
pend = N - M
plt.subplot(212)
while pin<pend:
	y [pin:pin+M] += w1*H
	plt.plot(np.arange(pin, pin+M), w, 'b', lw=1.5)
	pin += H
plt.plot(np.arange(0, N), y, 'r', lw=1.5)
plt.axis([0, N-H, 0, max(y)+.01])
plt.title('Blackman, M=201, H=50')

plt.tight_layout()
plt.savefig('window-overlap.png')
plt.show()

