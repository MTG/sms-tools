import matplotlib.pyplot as plt
import numpy as np

N = 500
k = 3
plt.figure(1)

s = np.exp(1j*2*np.pi*k/N*np.arange(-N/2, N/2))
plt.subplot(1, 2, 1)
plt.plot(np.arange(-N/2, N/2), np.real(s), lw=2)
plt.axvline(0, color='g', lw=2)
plt.axis([-N/2,N/2,-1,1])
plt.title ('cosine (even)')
plt.subplot(1, 2, 2)
plt.plot(np.arange(-N/2, N/2), np.imag(s), lw=2)
plt.axvline(0, color='g', lw=2)
plt.axis([-N/2,N/2,-1,1])
plt.title ('sine (odd)')


plt.tight_layout()
plt.savefig('even-odd.png')
plt.show()
