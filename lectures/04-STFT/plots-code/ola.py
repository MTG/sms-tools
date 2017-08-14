import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read

(fs, x) = read('../../../sounds/oboe-A4.wav')
M = 256
H = 128
start = int(.8*fs)

plt.figure(1)
x0 = x[start:start+3*M]/float(max(x))
plt.plot(x0)
plt.axis([0, 3*M, min(x0), max(x0)+5.5])

offset = 1.5
x1 = np.zeros(3*M)+offset
x1[0:M] += (x0[0:M] * np.hamming(M))
plt.plot(x1,'b')

offset = 2.5
x2 = np.zeros(3*M)+offset
x2[H:M+H] += (x0[H:M+H] * np.hamming(M))
plt.plot(x2,'b')

offset = 3.5
x2 = np.zeros(3*M)+offset
x2[H*2:M+H*2] += (x0[2*H:M+H*2] * np.hamming(M))
plt.plot(x2,'b')

offset = 4.5
x2 = np.zeros(3*M)+offset
x2[H*3:M+H*3] += (x0[3*H:M+H*3] * np.hamming(M))
plt.plot(x2,'b')

offset = 5.5
x2 = np.zeros(3*M)+offset
x2[H*4:M+H*4] += (x0[4*H:M+H*4] * np.hamming(M))
plt.plot(x2,'b')

plt.tight_layout()
plt.savefig('ola.png')
plt.show()
