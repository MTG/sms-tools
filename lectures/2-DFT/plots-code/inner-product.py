import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,1,1,1,-1,-1,-1,-1])
N = 8
mX = ()
pX = ()
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(x, 'b')
plt.plot(x,'x',color='b')
plt.axis([0,N-1,-1.5,1.5])
plt.title('x = [1,1,1,1,-1,-1,-1,-1]')
plt.xlabel('n')

for k in range(8):
  s = np.exp(-1j*2*np.pi*k/N*np.arange(N))
  X = sum(x*np.conjugate(s))
  mX = np.append(mX, np.abs(X))
  pX = np.append(pX, np.angle(X))

plt.subplot(3,1,2)
plt.plot(mX, 'r')
plt.plot(mX, 'x', color='r')
plt.title('$abs(<x,s_k>)$')
plt.xlabel('k')

plt.subplot(3,1,3)
plt.plot(pX, 'c')
plt.plot(pX, 'x', color='c')
plt.title('$angle(<x,s_k>)$')
plt.xlabel('k')

plt.show()
