import numpy as np
import matplotlib.pyplot as plt

p = 0
b= 10
a = -3
n = np.arange(-4,4,.1)
x = a * (n - p)**2 + b

plt.figure(1, figsize=(9.5, 5))

plt.plot(n,x,'r', lw=1.5)

n = np.arange(-4.5,4.5,1)
x = a * (n - p)**2 + b
plt.plot(n,x, 'x', color='b', markeredgewidth=1.5)
plt.axis([-4,4,-30,13])

plt.tight_layout()
plt.savefig('parabola.png')
plt.show()
