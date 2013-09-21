import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hamming, triang, blackmanharris

triangle = np.zeros (70)
triangle[1:32] = triang (31)
rectangle = np.zeros (70)
rectangle[1:32] = np.ones (31)
output = np.zeros (70)
output = np.convolve(triangle, rectangle)

plt.figure(1)
plt.subplot(3, 1, 1)
plt.plot(triangle)
plt.axis([0,69,-.1,1.1])
plt.title ('triangle')
plt.subplot(3, 1, 2)
plt.plot(rectangle)
plt.axis([0,69,-.1,1.1])
plt.title ('rectangle')
plt.subplot(3, 1, 3)
plt.plot(output)
plt.axis([0,69,-1,17])
plt.title ('triangle * rectangle')
plt.show()