import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window

f = 20000.0 * np.arange(0,100)/100.0
m = 2595 * np.log10(1+f/700)


plt.figure(1, figsize=(9, 6))
plt.plot(f, m)
plt.grid()
plt.axis([0, 20000, min(m), max(m)])
plt.xlabel('Hertz scale')
plt.ylabel('Mel scale')

plt.tight_layout()
plt.savefig('mel-scale.png')
plt.show()
