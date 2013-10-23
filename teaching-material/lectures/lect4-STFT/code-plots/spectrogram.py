from pylab import *
from scipy.io.wavfile import read
import numpy as np

(fs, x) = read('piano.wav')
figure(1)
subplot(2,1,1)
specgram(x, NFFT=256, window= np.hamming(256), Fs=fs, noverlap=128)
axis([0,size(x)/fs,0,fs/4])
title('N = 256, window = hamming(256), H = 128')
subplot(2,1,2)
specgram(x, NFFT=1024, window= np.hamming(1024), Fs=fs, noverlap=128)
axis([0,size(x)/fs,0,fs/4])
title('N = 1024, window = hamming(1024), H = 128')
show()