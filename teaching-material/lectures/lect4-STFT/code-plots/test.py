from pylab import *
from scipy.io.wavfile import read

(fs, x) = read('piano.wav')
figure(1)
specgram(x, NFFT=256, Fs=fs, noverlap=128)
axis([0,size(x)/fs,0,fs/2])
figure(2)
specgram(x, NFFT=1024, Fs=fs, noverlap=128)
axis([0,size(x)/fs,0,fs/2])
show()