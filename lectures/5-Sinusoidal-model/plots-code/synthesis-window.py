import numpy as np
from scipy.signal import hamming, triang, blackmanharris
import matplotlib.pyplot as plt

Ns = 512                                                # FFT size for synthesis (even)
hNs = Ns/2                                              # half of synthesis FFT size
H = Ns/4 
sw = np.zeros(Ns)                                       # initialize synthesis window
ow = triang(2*H);                                       # triangular window
sw[hNs-H:hNs+H] = ow                                    # add triangular window
bh = blackmanharris(Ns)                                 # blackmanharris window

plt.figure(1)
plt.subplot(3,1,1)
plt.plot(bh,'b')
plt.axis([0, Ns,0,1])
plt.title("blackmanharris window")

plt.subplot(3,1,2)
plt.plot(sw,'b')
plt.axis([0, Ns, 0, 1])
plt.title("triangular window")

sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
plt.subplot(3,1,3)
plt.plot(sw,'b')
plt.axis([0, Ns, 0, 1])
plt.title("triangular window / blackmanharris window")

f = 100.0
fs = 1000.0
k = Ns*f/fs 

x = np.cos(2*np.pi*k/Ns*np.arange(Ns)) * blackmanharris(Ns)

plt.figure(2)
plt.subplot(3,1,1)
plt.plot(x,'b')
plt.axis([0, Ns,-1,1])
plt.title("x")

plt.subplot(3,1,2)
plt.plot(sw,'b')
plt.axis([0, Ns,0,1])
plt.title("synthesis window")

plt.subplot(3,1,3)
plt.plot(x*sw,'b')
plt.axis([0, Ns,-1,1])
plt.title("x * synthesis window")

plt.show()