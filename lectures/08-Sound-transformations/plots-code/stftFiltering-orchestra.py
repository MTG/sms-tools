import numpy as np
import time, os, sys
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/transformations/'))
import utilFunctions as UF
import stftTransformations as STFTT
import stft as STFT

(fs, x) = UF.wavread('../../../sounds/orchestra.wav')
w = np.hamming(2048)
N = 2048
H = 512
# design a band stop filter using a hanning window
startBin = int(N*500.0/fs)
nBins = int(N*2000.0/fs)
bandpass = (np.hanning(nBins) * 65.0) - 60
filt = np.zeros(N//2+1)-60
filt[startBin:startBin+nBins] = bandpass
y = STFTT.stftFiltering(x, fs, w, N, H, filt)
mX,pX = STFT.stftAnal(x, w, N, H)
mY,pY = STFT.stftAnal(y, w, N, H)

plt.figure(1, figsize=(12, 9))
plt.subplot(311)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(mX[0,:].size)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
plt.title('mX (orchestra.wav)')
plt.autoscale(tight=True)

plt.subplot(312)
plt.plot(fs*np.arange(mX[0,:].size)/float(N), filt, 'k', lw=1.3)
plt.axis([0, fs/2, -60, 7])
plt.title('filter shape')

plt.subplot(313)
numFrames = int(mY[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(mY[0,:].size)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mY))
plt.title('mY')
plt.autoscale(tight=True)

plt.tight_layout()
UF.wavwrite(y, fs, 'orchestra-stft-filtering.wav')
plt.savefig('stftFiltering-orchestra.png')
plt.show()
