# example of analysing, synthesizing, and transforming the cello-phrase sound 

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming, triang, blackmanharris
import sys, os, functools, time, copy
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/transformations/'))
import hpsModel as HPS
import harmonicModel as HM
import stft as STFT
import sineModel as SM
import hpsTransformations as HPST
import harmonicTransformations as HT
import utilFunctions as UF

# read the sound
(fs, x) = UF.wavread(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../sounds/cello-phrase.wav'))

# plot the waveform
plt.figure(1, figsize=(16, 4.5))
plt.plot(np.arange(x.size)/float(fs), x, 'b')
plt.axis([0,x.size/float(fs),min(x),max(x)])
plt.xlabel('time (sec)')
plt.ylabel('amplitude')
plt.title('waveform')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-waveform.png')

# compute the STFT
w = np.blackman(801)
N = 2048
H = 128
mX, pX = STFT.stftAnal(x, fs, w, N, H)

# plot the spectrogram
plt.figure(2, figsize=(16, 4.5))
maxplotfreq = 5000.0
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                          
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-spectrogram.png')

# compute the FO and the harmonics
t = -97
minf0 = 310
maxf0 = 450
f0et = 4
nH = 70
harmDevSlope = 0.01
Ns = H*4
minSineDur = .3
hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
hfreqt = copy.copy(hfreq)
hfreqt[:,1:] = 0
yf0 = 4*SM.sineModelSynth(hfreqt, hmag, hphase, Ns, H, fs)
yh = SM.sineModelSynth(hfreq, hmag, hphase, Ns, H, fs)
UF.wavwrite(yf0, fs, 'cello-phrase-f0.wav')
UF.wavwrite(yh, fs, 'cello-phrase-harmonics.wav')

# plot the F0 on top of the spectrogram
plt.figure(3, figsize=(16, 4.5))
maxplotfreq = 5000.0
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms[:,0]==0] = np.nan
numFrames = int(mX[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                   
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.plot(frmTime, harms[:,0], linewidth=3, color='0')
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram + fundamental frequency')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-f0.png')

# plot the harmonics on top of the spectrogram
plt.figure(4, figsize=(16, 4.5))
maxplotfreq = 5000.0
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
binFreq = fs*np.arange(N*maxplotfreq/fs)/N
plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]))
plt.plot(frmTime, harms, linewidth=1.5, color='0.1')
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram + harmonic frequencies')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-harmonics.png')

# subtract the harmonics from the sound to obtain the residual
xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
mXr, pXr = STFT.stftAnal(xr, fs, hamming(Ns), Ns, H)
UF.wavwrite(xr, fs, 'cello-phrase-residual.wav')

# plot the spectrogram of the residual
plt.figure(5, figsize=(16, 4.5))
maxplotfreq = 5000.0
numFrames = int(mXr[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                          
plt.pcolormesh(frmTime, binFreq, np.transpose(mXr[:,:N*maxplotfreq/fs+1]))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram of residual')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-residual.png')

# compute the harmonic plus stochastic model
stocf = .5
hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)
y, yh, yst = HPS.hpsModelSynth(hfreq, hmag, np.array([]), mYst, Ns, H, fs)
UF.wavwrite(yst, fs, 'cello-phrase-stochastic.wav')
UF.wavwrite(yh, fs, 'cello-phrase-harmonic.wav')
UF.wavwrite(y, fs, 'cello-phrase-synthesis.wav')

# plot the spectrogram of the stochastic component
plt.figure(6, figsize=(16, 4.5))
maxplotfreq = 5000.0
numFrames = int(mYst[:,0].size)
Nst = 2*int(mYst[0,:].size)
lastbin = int(Nst*maxplotfreq/fs)
maxplotfreq = fs*(lastbin-1)/Nst
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(lastbin)/Nst                          
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:lastbin]))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram of stochastic model')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-stochastic.png')

# plot the spectrogram of the stochastic component and the harmonics
plt.figure(7, figsize=(16, 4.5))
maxplotfreq = 5000.0
numFrames = int(mYst[:,0].size)
Nst = 2*int(mYst[0,:].size)
lastbin = int(Nst*maxplotfreq/fs)
maxplotfreq = fs*(lastbin-1)/Nst
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(lastbin)/Nst                          
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:lastbin]))
numFrames = int(harms[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs) 
harms = hfreq*np.less(hfreq,maxplotfreq)
harms[harms==0] = np.nan
plt.plot(frmTime, harms, color='0.2', linewidth=1.5)
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram of stochastic model + harmonic frequencies')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-harmonic-plus-stochastic.png')

# plot the note onsets on the spectrogram of the stochastic component and the harmonics
onsets = np.array([0.08, .679, 1.18, 1.729, 2.993, 3.609, 4.7, 5.902]) 
plt.figure(8, figsize=(16, 4.5))
maxplotfreq = 5000.0
numFrames = int(mYst[:,0].size)
Nst = 2*int(mYst[0,:].size)
lastbin = int(Nst*maxplotfreq/fs)
maxplotfreq = fs*(lastbin-1)/Nst
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(lastbin)/Nst                          
plt.pcolormesh(frmTime, binFreq, np.transpose(mYst[:,:lastbin]))
numFrames = int(harms[:,0].size)
harms = hfreq*np.less(hfreq, maxplotfreq)
harms[harms==0] = np.nan
frmTime = H*np.arange(numFrames)/float(fs) 
plt.plot(frmTime, harms, color='0.2', linewidth=1)
plt.vlines(onsets, 0, maxplotfreq, color='r', lw=3, linestyles='dashed')
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram of stochastic model + harmonic frequencies + note onsets')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-note-onsets.png')

# transfrom the pitch of the sound
freqScaling = np.array([0,.94, .61, .94, .679, .94, 1.144, .94, 1.18, 1.12, 1.647, 1.12, 1.72, 1, 2.928, 1, 2.98, 1.12, 3.524, 1.12, 3.62, 1.32, 4.55, 1.32, 4.7, 1.65, 5.84, 1.65, 5.93, 1.967, 8.483, 1.967]) 
freqStretching = np.array([])
timbrePreservation = 1
hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)
yh = SM.sineModelSynth(hfreqt, hmagt, np.array([]), Ns, H, fs)
length = min(yh.size, xr.size) 
y = yh[:length] + xr[:length]
mY, pY = STFT.stftAnal(y, fs, w, N, H)
UF.wavwrite(y, fs, 'cello-phrase-pitch-transformation.wav')

# plot the spectrogram of the transformed sound
plt.figure(9, figsize=(16, 4.5))
maxplotfreq = 5000.0
numFrames = int(mY[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                          
plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:,:N*maxplotfreq/fs+1]))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram of transformed sound')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-pitch-transformation.png')

# make some weird transformations to the sound
freqScaling = np.array([0, 1, .61, 1, .679, 1, 1.144, 1, 1.18, .8, 1.647, .8, 1.72, 1.5, 3.62, .5, 4.55, .5, 4.7, 1.5, 5.84, 1.5, 5.902, 1, 8.483, 1]) 
freqStretching = np.array([0, 1, .61, 1, .679, 1, 1.144, 1, 1.18, 1.05, 1.647, 1.05, 1.72, 1, 2.928, 1, 2.98, 1, 3.524, 1,  3.609, 2, 4.573, 2, 4.7, .9, 5.85, .9, 5.902, .7, 8.483, 4]) 
timbrePreservation = 0
hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)
timeScaling = np.array([0, 0, .61, .61+.5, .679, .679+.5, 1.144, 1.144+.3, 1.18, 1.18+.3, 1.647, 1.647+.2, 1.72, 1.72+.2, 2.928, 2.928-.2, 2.98, 2.98-.2, 3.524, 3.524, 3.609, 3.609, 4.573, 4.573, 4.7, 4.7, 5.8-.4, 5.8-.4, 5.902-.4, 5.902-.4, 8.483, 8.483]) 
yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreqt, hmagt, mYst, timeScaling)
y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)
mY, pY = STFT.stftAnal(y, fs, w, N, H)
UF.wavwrite(y, fs, 'cello-phrase-weird-transformation.wav')

# plot the spectrogram of the transformed sound
plt.figure(10, figsize=(16, 4.5))
maxplotfreq = 5000.0
numFrames = int(mY[:,0].size)
frmTime = H*np.arange(numFrames)/float(fs)                             
binFreq = fs*np.arange(N*maxplotfreq/fs)/N                          
plt.pcolormesh(frmTime, binFreq, np.transpose(mY[:,:N*maxplotfreq/fs+1]))
plt.xlabel('time (sec)')
plt.ylabel('frequency (Hz)')
plt.title('spectrogram of transformed sound')
plt.autoscale(tight=True)
plt.tight_layout()
plt.savefig('cello-phrase-weird-transformation.png')


