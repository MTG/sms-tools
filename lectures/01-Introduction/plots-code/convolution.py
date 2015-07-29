import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming, triang, blackmanharris

triangle = np.zeros (70)
triangle[1:32] = triang (31)
rectangle = np.zeros (70)
rectangle[1:32] = np.ones (31)
output = np.zeros (70)
output = np.convolve(triangle, rectangle)

plt.figure(1, figsize=(9.5, 5.5))
plt.subplot(3, 1, 1)
plt.plot(triangle, lw=2)
plt.axis([0,69,-.1,1.1])
plt.title ('x1 (triangle)')
plt.subplot(3, 1, 2)
plt.plot(rectangle, lw=2)
plt.axis([0,69,-.1,1.1])
plt.title ('x2 (rectangle)')
plt.subplot(3, 1, 3)
plt.plot(output, lw=2)
plt.axis([0,69,-1,17])
plt.title ('y = x1 * x2')

plt.tight_layout()
plt.savefig('convolution.png')
plt.show()
