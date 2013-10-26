import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))

import smsWavplayer as wp
from scipy import signal
import matplotlib.pyplot as plt
import importlib
import inspect
import sys
import pprint
import wave
import pyaudio
import os, copy
from scipy.io.wavfile import write
from scipy.io.wavfile import read
import math
from numpy.fft import fft, ifft

def stft2(x, fs, w, N, H):
    ''' Analysis/synthesis of a sound using the short-time fourier transform
    x: input array sound, w: analysis window, N: FFT size, H: hop size
    returns y: output array sound 
    YSpec: The STFT of x (Only the half spectrum is stored)
    frmTime: The time values at which the STFT was computed
    binFreq: The frequency values of the bins at which the STFT was computed
    If T = len(frmTime) and F = len(binFreq), YSpec is a FxT sized matrix '''
    
    hN = N/2                                                # size of positive spectrum
    hM1 = int(math.floor((w.size+1)/2))                     # Ceil of half analysis window size
    hM2 = int(math.floor(w.size/2))                         # Floor of half analysis window size
    pin = hM1                                               # initialize sound pointer in middle of analysis window       
    pend = x.size-max(hM1,H)                                # last sample to start a frame
    fftbuffer = np.zeros(N)                                 # initialize buffer for FFT
    yw = np.zeros(w.size)                                   # initialize output sound frame
    y = np.zeros(x.size)                                    # initialize output array
    w = w / sum(w)                                          # normalize analysis window
    numFrames = int(math.floor(pend/float(H)))
    frmNum = 0
    frmTime = []
    binFreq = np.arange(N/2)*float(fs)/N                    # The bin frequencies
    while pin<pend:                                         # while sound pointer is smaller than last sample    
        frmTime.append(pin/float(fs))
        #-----analysis-----             
        xw = x[pin-hM1:pin+hM2]*w                             # window the input sound
        fftbuffer = np.zeros(N)                               # clean fft buffer
        fftbuffer[:hM1] = xw[hM2:]                            # zero-phase window in fftbuffer
        fftbuffer[N-hM2:] = xw[:hM2]        
        X = fft(fftbuffer)                                    # compute FFT
        mX = 20 * np.log10( abs(X[:hN]) )                     # magnitude spectrum of positive frequencies in dB     
        pX = np.unwrap( np.angle(X[:hN]) )                    # unwrapped phase spectrum of positive frequencies
        
        if frmNum == 0:                                       # Accumulate and store STFT
            YSpec = np.transpose(np.array([X[:hN]]))
        else:
            YSpec = np.hstack((YSpec,np.transpose(np.array([X[:hN]]))))
        #-----synthesis-----
        Y = np.zeros(N, dtype = complex)                      # clean output spectrun
        Y[:hN] = 10**(mX/20) * np.exp(1j*pX)                  # generate positive frequencies
        Y[hN+1:] = 10**(mX[:0:-1]/20) * np.exp(-1j*pX[:0:-1]) # generate negative frequencies
        fftbuffer = np.real(ifft(Y) )                         # compute inverse FFT
        yw[:hM2] = fftbuffer[N-hM2:]                          # undo zero-phase window
        yw[hM2:] = fftbuffer[:hM1]
        y[pin-hM1:pin+hM2] += H*yw                            # overlap-add
        pin += H                                              # advance sound pointer
        frmNum += 1
    
    frmTime = np.array(frmTime)                               # The time at the centre of the frames
    
    return (y, YSpec, frmTime, binFreq)

# example call of stft function
if __name__ == '__main__':
    (fs, x) = wp.wavread('../../sounds/oboe.wav')
    w = np.hamming(511)
    N = 1024
    H = 256
    (yout, YSpec, frmTime, binFreq) = stft2(x,fs,w,N,H)
    plt.pcolormesh(frmTime,binFreq,20*np.log10(abs(YSpec)))
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(Hz)')
    plt.title('oboe')
    plt.autoscale(tight=True)
    plt.show()

