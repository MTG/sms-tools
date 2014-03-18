import numpy as np
import time, os, sys
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/transformations/'))

import waveIO as WIO
import stftFiltering as STFTF
import stftAnal as STFT

(fs, x) = WIO.wavread('../../../sounds/orchestra.wav')
w = np.hamming(2048)
N = 2048
H = 512
filter = np.array([0, -40, 200, -40, 300, 0, 600, 0, 700,-40, 1500, -40, 1600, 0, 2500, 0, 2800, -40, 22050, -60])
filt = np.interp(np.arange(N/2), (N/2)*filter[::2]/filter[-2], filter[1::2])
y = STFTF.stftFiltering(x, fs, w, N, H, filter)
mX,pX = STFT.stftAnal(x, fs, w, N, H)
mY,pY = STFT.stftAnal(y, fs, w, N, H)

plt.figure(1)
plt.subplot(311)
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mX))
plt.title('mX (orchestra.wav)')
plt.autoscale(tight=True)

plt.subplot(312)
plt.plot(fs*np.arange(N/2)/float(N), filt, 'k', lw=1.3)
plt.axis([0, fs/2, -60, 2])
plt.title('filter shape')

plt.subplot(313)
numFrames = int(mY[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = np.arange(N/2)*float(fs)/N                         
plt.pcolormesh(frmTime, binFreq, np.transpose(mY))
plt.title('mY')
plt.autoscale(tight=True)

plt.show()
