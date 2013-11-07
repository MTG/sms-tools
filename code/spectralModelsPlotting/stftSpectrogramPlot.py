import numpy as np
import time, os, sys

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../basicFunctions/'))

import smsWavplayer as wp
from scipy import signal
import matplotlib.pyplot as plt
import math
from numpy.fft import fft

def stftSpectrogramPlot(x, fs, w, N, H, minFreq, maxFreq):  
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
    firstBin = N*minFreq/float(fs)
    lastBin = N*maxFreq/float(fs)
    binFreq = np.arange(firstBin,lastBin)*float(fs)/N       # The bin frequencies
    while pin<pend:                                         # while sound pointer is smaller than last sample    
        frmTime.append(pin/float(fs))         
        xw = x[pin-hM1:pin+hM2]*w                             # window the input sound
        fftbuffer = np.zeros(N)                               # clean fft buffer
        fftbuffer[:hM1] = xw[hM2:]                            # zero-phase window in fftbuffer
        fftbuffer[N-hM2:] = xw[:hM2]        
        X = fft(fftbuffer)                                    # compute FFT
        mX = 20 * np.log10( abs(X[firstBin:lastBin]) )        # magnitude spectrum of positive frequencies in dB     
        pX = np.unwrap( np.angle(X[firstBin:lastBin]) )       # unwrapped phase spectrum of positive frequencies
        if frmNum == 0:                                       # Accumulate and store STFT
            YSpec = np.transpose(np.array([mX]))
        else:
            YSpec = np.hstack((YSpec,np.transpose(np.array([mX]))))
        pin += H
        frmNum += 1
    frmTime = np.array(frmTime)                               # The time at the centre of the frames
    plt.pcolormesh(frmTime,binFreq,YSpec)
    plt.plot(3,2,'ro')
    plt.xlabel('Time(s)')
    plt.ylabel('Frequency(Hz)')
    plt.autoscale(tight=True)
    plt.show()
    return YSpec

# example call of stftPlot function
if __name__ == '__main__':
    (fs, x) = wp.wavread('../../sounds/oboe-A4.wav')
    w = np.hamming(511)
    N = 1024
    H = 256
    minFreq = 0
    maxFreq = fs/10.0
    YSpec = stftSpectrogramPlot(x,fs,w,N,H,minFreq,maxFreq)
   
