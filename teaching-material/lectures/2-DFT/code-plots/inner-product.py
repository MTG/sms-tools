import numpy as np

np.set_printoptions(precision=4, suppress=True)
x = np.array([1,-1,1,-1])
print 'x = {}'.format(x)
N = 4
for k in range(N):
	s = np.exp(1j*2*np.pi*k/N*np.arange(N))
	print 's{0} = {1}'.format(k, s)
	X = sum(x*np.conjugate(s))
	print '<x,s{0}> = {1}'.format(k,X)