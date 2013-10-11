from pylab import *
from scipy.io.wavfile import read

(fs, x) = read('piano.wav')
figure(1)
subplot(2,1,1)
specgram(x, NFFT=256, Fs=fs, noverlap=128)
axis([0,size(x)/fs,0,fs/4])
subplot(2,1,2)
specgram(x, NFFT=1024, Fs=fs, noverlap=128)
axis([0,size(x)/fs,0,fs/4])
show()