import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, hanning, triang, blackmanharris, resample
from scipy.fftpack import fft, ifft, fftshift
import sys, os, functools, time, math
from scipy.interpolate import interp1d
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../software/models/'))
import sineModel as SM 
import stft as STFT
import sineModel as SM
import utilFunctions as UF

(fs, x) = UF.wavread('../../../sounds/mridangam.wav')
x1 = x[:int(1.49*fs)]
w = np.hamming(801)
N = 2048
t = -90
minSineDur = .005
maxnSines = 150
freqDevOffset = 20
freqDevSlope = 0.02
Ns = 512
H = Ns//4
sfreq, smag, sphase = SM.sineModelAnal(x1, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
timeScale = np.array([.01, .0, .03, .03, .335, .8, .355, .82, .671, 1.0, .691, 1.02, .858, 1.1, .878, 1.12, 1.185, 1.8, 1.205, 1.82, 1.49, 2.0])          
L = sfreq[:,0].size                                    # number of input frames
maxInTime = max(timeScale[::2])                      # maximum value used as input times
maxOutTime = max(timeScale[1::2])                    # maximum value used in output times
outL = int(L*maxOutTime/maxInTime)                     # number of output frames
inFrames = L*timeScale[::2]/maxInTime                # input time values in frames
outFrames = outL*timeScale[1::2]/maxOutTime          # output time values in frames
timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)    # interpolation function
indexes = timeScalingEnv(np.arange(outL))              # generate frame indexes for the output
ysfreq = sfreq[int(round(indexes[0])),:]                    # first output frame
ysmag = smag[int(round(indexes[0])),:]                      # first output frame
for l in indexes[1:]:                                  # generate frames for output sine tracks
	ysfreq = np.vstack((ysfreq, sfreq[int(round(l)),:]))  
	ysmag = np.vstack((ysmag, smag[int(round(l)),:])) 

mag1 = np.sum(10**(smag/20), axis=1)
mag2 = np.sum(10**(ysmag/20), axis=1)
mag1 = 20*np.log10(mag1)
mag2 = 20*np.log10(mag2)

plt.figure(1, figsize=(9, 7))
maxplotfreq = 4000.0
plt.subplot(3,1,1)
plt.plot(H*indexes/float(fs), H*np.arange(outL)/float(fs), color='k', lw=1.5)
plt.autoscale(tight=True)
plt.xlabel('input times')
plt.ylabel('output times')
plt.title('output scaling')   

plt.subplot(3,1,2)
plt.plot(H*np.arange(mag1.size)/float(fs), mag1, color='k', lw=1.5)
plt.autoscale(tight=True)
plt.title('input magnitude sines')  


plt.subplot(3,1,3)
plt.plot(H*np.arange(mag2.size)/float(fs), mag2, color='k', lw=1.5)
plt.autoscale(tight=True)
plt.title('output magnitude sines') 


plt.tight_layout()
plt.savefig('sineModelTimeScale-functions.png')
plt.show()

