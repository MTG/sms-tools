import numpy as np
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import time

timeDFT = np.array([])
timeFFT = np.array([])
Ns = 2**np.arange(7,15)
for N in Ns:
	x = np.random.rand(N)
	X = np.array([])
	str_time = time.time()
	for k in range(N):
		s = np.exp(1j*2*np.pi*k/N*np.arange(N))
		X = np.append(X, sum(x*np.conjugate(s)))
	timeDFT = np.append(timeDFT, time.time()-str_time)

x = np.random.rand(120)
X = fft(x)
for N in Ns:
	x = np.random.rand(N)
	str_time = time.time()
	X = fft(x)
	timeFFT = np.append(timeFFT, time.time()-str_time)

plt.figure(1, figsize=(9.5, 7))
plt.subplot(2,1,1)
plt.plot(timeDFT, 'b', lw=1.5)
plt.title('DFT compute time')
plt.xlabel('N')
plt.ylabel('seconds')
plt.xticks(np.arange(len(Ns)), Ns)
plt.subplot(2,1,2)
plt.plot(timeFFT, 'b', lw=1.5)
plt.title('FFT compute time')
plt.xlabel('N')
plt.ylabel('seconds')
plt.xticks(np.arange(len(Ns)), Ns)

plt.tight_layout()
plt.savefig('dft-fft.png')
plt.show()
