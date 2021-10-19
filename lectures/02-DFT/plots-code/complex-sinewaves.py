import matplotlib.pyplot as plt
import numpy as np

N = 8
fig, axs = plt.subplots(int(N/2),2,figsize=(9.5, 6))
axs = axs.ravel()
for k in range(N):
	s = np.exp(-1j*2*np.pi*k/N*np.arange(N))
	axs[k].plot(np.real(s), 'b-x', lw=1.5)
	axs[k].axis([0,N-1,-1.5,1.5])
	axs[k].set_title(r"$s^{*}_{%s}$"%(k), fontsize=18)
	axs[k].plot(np.imag(s), 'g-x', lw=1.5)
	axs[k].axis([0,N-1,-1.5,1.5])

plt.tight_layout()
plt.savefig('complex-sinewaves.png')
plt.show()
