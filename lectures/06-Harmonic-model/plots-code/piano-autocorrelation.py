import matplotlib.pyplot as plt
import numpy as np
import math
import time, os, sys
import essentia.standard as ess

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))

import utilFunctions as UF
(fs, x) = UF.wavread('../../../sounds/piano.wav')
start = 13860
M = 800 
xp = x[start:start+M]/float(max(x[start:start+M]))
r = ess.AutoCorrelation(normalization = 'standard')(xp)
r = r / max(r)
peaks = ess.PeakDetection(threshold =.11, interpolate = False, minPosition = .01)(r)

plt.figure(1, figsize=(9, 7))
plt.subplot(211)
plt.plot(np.arange(M)/float(fs), xp, lw=1.5)
plt.axis([0, (M-1)/float(fs), min(xp), max(xp)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.title('x (piano.wav)')

plt.subplot(212)
plt.plot(np.arange(M)/float(fs), r, 'r', lw=1.5)
plt.plot(peaks[0]*(M-1)/float(fs),peaks[1], 'x', color='k', markeredgewidth=1.5)
plt.axis([0, (M-1)/float(fs), min(r), max(r)])
plt.title('autocorrelation function + peaks')
plt.xlabel('lag time (sec)')
plt.ylabel('correlation')

plt.tight_layout()
plt.savefig('piano-autocorrelation.png')
plt.show()