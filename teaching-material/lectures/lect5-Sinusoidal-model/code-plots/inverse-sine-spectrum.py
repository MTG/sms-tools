import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift, ifft
import matplotlib.pyplot as plt

plt.figure(1)

M = 255
N = 1024
f = 100.0
fs = 1000.0
k = N*f/fs
hN = N/2                                               
hM = (M+1)/2    

x = np.cos(2*np.pi*k/N*np.arange(M)) * signal.hamming(M)
fftbuffer = np.zeros(N)                           
fftbuffer[:hM] = x[hM-1:]                        
fftbuffer[N-hM+1:] = x[:hM-1]        
X = fft(fftbuffer)                   
mX = 20 * np.log10( abs(X[:hN]) )                      
pX = np.unwrap( np.angle(X[:hN]) )                   
lb = round(k-(2*N/float(M)))
rb = round(k+(2*N/float(M)))

plt.subplot(2,2,1)
plt.plot(mX-max(mX), 'b')
plt.axis([0,hN,-100,0])
plt.axvline(lb, color='r')
plt.axvline(rb, color='r')
plt.xlabel('Hamming')

mask = np.zeros(hN)
mask[lb:rb] = np.ones(rb-lb)
mX *= mask
Y = np.zeros(N, dtype = complex) 
y =  np.zeros(M)
Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                  
Y[hN+1:] = 10**(mX[:0:-1]/20) * np.exp(-1j*pX[:0:-1]) 
fftbuffer = np.real( ifft(Y) )                        
y[:hM-1] = fftbuffer[N-hM+1:]  
y[hM-1:] = fftbuffer[:hM]

plt.subplot(2,2,2)
plt.plot(y[0:M], 'r')
plt.plot(x[0:M], 'b')
plt.axis([0,M,-1,1])
plt.xlabel('Hamming inverse')

x = np.cos(2*np.pi*k/N*np.arange(M)) * signal.blackmanharris(M)
fftbuffer = np.zeros(N)              
fftbuffer[:hM] = x[hM-1:]                         
fftbuffer[N-hM+1:] = x[:hM-1]        
X = fft(fftbuffer)                         
mX = 20 * np.log10( abs(X[:hN]) )                         
pX = np.unwrap( np.angle(X[:hN]) )            
lb = round(k-(4*N/float(M)))
rb = round(k+(4*N/float(M)))

plt.subplot(2,2,3)
plt.plot(mX-max(mX), 'b')
plt.axis([0,hN,-100,0])
plt.axvline(lb, color='r')
plt.axvline(rb, color='r')
plt.xlabel('BlackmanHarris')

plt.subplot(2,2,4)
mask = np.zeros(hN)
mask[lb:rb] = np.ones(rb-lb)
mX *= mask
Y = np.zeros(N, dtype = complex) 
y =  np.zeros(M)
Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                  
Y[hN+1:] = 10**(mX[:0:-1]/20) * np.exp(-1j*pX[:0:-1]) 
fftbuffer = np.real( ifft(Y) )                        
y[:hM-1] = fftbuffer[N-hM+1:]  
y[hM-1:] = fftbuffer[:hM]
plt.plot(y[0:M], 'r')
plt.plot(x[0:M], 'b')
plt.axis([0,M,-1,1])
plt.xlabel('BlackmanHarris inverse')

plt.show()
