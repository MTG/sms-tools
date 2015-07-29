import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,1,1,1,-1,-1,-1,-1])
N = 8
mX = ()
pX = ()
plt.figure(1, figsize=(9.5, 6))

plt.subplot(3,1,1)
plt.plot(x,marker='x',color='b', lw=1.5)
plt.axis([0,N-1,-1.5,1.5])
plt.title('x = [1,1,1,1,-1,-1,-1,-1]', fontsize=18)

for k in range(8):
  s = np.exp(1j*2*np.pi*k/N*np.arange(N))
  X = sum(x*np.conjugate(s))
  mX = np.append(mX, np.abs(X))
  pX = np.append(pX, np.angle(X))

plt.subplot(3,1,2)
plt.plot(mX, marker='x', color='r', lw=1.5)
plt.title('$abs(<x,s_k>)$', fontsize=18)

plt.subplot(3,1,3)
plt.plot(pX, marker='x', color='c', lw=1.5)
plt.title('$angle(<x,s_k>)$', fontsize=18)

plt.tight_layout()
plt.savefig('inner-product.png')
plt.show()
