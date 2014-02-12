import matplotlib.pyplot as plt
import numpy as np

N = 8
plt.figure(1)
for k in range(N):
	s = np.exp(-1j*2*np.pi*k/N*np.arange(N))
	plt.subplot(N/2, 2, k+1)
	plt.plot(np.real(s), 'b')
	plt.axis([0,N-1,-1.5,1.5])
	plt.title(r"$\overline{s}_{%s}$"%(k))
	plt.subplot(N/2, 2, k+1)
	plt.plot(np.imag(s), 'g')
	plt.axis([0,N-1,-1.5,1.5])
plt.show()
