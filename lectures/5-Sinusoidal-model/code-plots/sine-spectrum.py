import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt

plt.figure(1)

plt.subplot(2,1,1)
window = signal.hamming(51)
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
plt.plot(freq, response)
plt.axis([-0.5, 0.5, -90, 0])
plt.title("Hamming window")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Frequency")

plt.subplot(2,1,2)
window = signal.blackmanharris(51)
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
plt.plot(freq, response)
plt.axis([-0.5, 0.5, -90, 0])
plt.title("BlackmanHarris window")
plt.ylabel("Amplitude (dB)")
plt.xlabel("Frequency")


plt.show()
