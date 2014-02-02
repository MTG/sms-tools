import matplotlib.pyplot as plt
import numpy as np

N = 500
k = 3
plt.figure(1)

s = np.exp(1j*2*np.pi*k/N*np.arange(-N/2, N/2))
plt.plot(np.arange(-N/2, N/2), np.real(s), 'b', label="real")
plt.plot(np.arange(-N/2, N/2), np.imag(s),'g', label="imaginary")
plt.xlabel('time')
plt.ylabel('amplitude')
plt.axis([-N/2,N/2,-1,1])
plt.legend()
plt.show()