import matplotlib.pyplot as plt
import numpy as np
import math
import time, os, sys
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/utilFunctions/'))

import waveIO as WIO
(fs, x) = WIO.wavread('../../../sounds/oboe-A4.wav')

M = 500
start = .8*fs   
xp = x[start:start+M]/float(max(x[start:start+M]))
z = ess.AutoCorrelation(normalization = 'standard')(xp)
zn = z / max(z)
peaks = ess.PeakDetection(threshold =.2, interpolate = False, minPosition = .01)(zn)

plt.figure(1)
plt.subplot(211)
plt.plot(np.arange(M)/float(fs), xp)
plt.axis([0, (M-1)/float(fs), min(xp), max(xp)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.title('x=wavread(oboe-A4.wav)')

plt.subplot(212)
plt.plot(np.arange(M)/float(fs), zn, 'r')
plt.plot(peaks[0]*(M-1)/float(fs),peaks[1], 'x', color='k')
plt.axis([0, (M-1)/float(fs), min(zn), max(zn)])
plt.title('Z = autocorrelation function + peaks')
plt.xlabel('lag time (sec)')
plt.ylabel('correlation')

plt.show()