# example of time scaling a sound using the harmonic plus stochastic model

import numpy as np
from scipy.signal import hamming, triang, blackmanharris
import sys, os, functools, time, math
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/models/'))
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../software/transformations/'))
import hpsModel as HPS
import hpsTransformations as HPST
import harmonicTransformations as HT
import utilFunctions as UF

# ------- analysis parameters -------------------

# ------- analysis parameters -------------------

# input sound (monophonic with sampling rate of 44100)
(fs, x) = UF.wavread('../sounds/sax-phrase-short.wav') 

# analysis window size 
M = 601

# analysis window type (rectangular, hanning, hamming, blackman, blackmanharris)	
w = np.blackman(M) 

# fft size (power of two, bigger or equal than M)
N = 1024             

# magnitude threshold of spectral peaks
t = -100  

# minimun duration of sinusoidal tracks
minSineDur = .1

# maximum number of harmonics
nH = 100  

# minimum fundamental frequency in sound
minf0 = 350 

# maximum fundamental frequency in sound
maxf0 = 700 

# maximum error accepted in f0 detection algorithm                                    
f0et = 5                                        

# allowed deviation of harmonic tracks, higher harmonics have higher allowed deviation
harmDevSlope = 0.01 

# decimation factor used for the stochastic approximation
stocf = .2  

# size of fft used in synthesis
Ns = 512

# hop size (has to be 1/4 of Ns)
H = 128


# --------- analysis -----------------

# harmonic plus stochastic analysis
hfreq, hmag, hphase, mYst = HPS.hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf)

# --------- transformation -----------------

# scale harmonic frequencies
freqScaling = np.array([0, 1.2, 1.18, 1.2, 1.89, 1.2, 2.01, 1.2, 2.679, .7, 3.146, .7])
freqStretching = np.array([])
timbrePreservation = 1
hfreqt, hmagt = HT.harmonicFreqScaling(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)

# time scale the sound
timeScaling = np.array([0, 0, 0.073, 0.073, 0.476, 0.476-0.2, 0.512, .512-0.2, 0.691, 0.691+0.2, 1.14, 1.14, 1.21, 1.21, 1.87, 1.87-0.4, 2.138, 2.138-0.4, 2.657, 2.657+.8, 2.732, 2.732+.8, 2.91, 2.91+.7, 3.146, 3.146+.7])
yhfreq, yhmag, ystocEnv = HPST.hpsTimeScale(hfreqt, hmagt, mYst, timeScaling)

# --------- synthesis -----------------

y, yh, yst = HPS.hpsModelSynth(yhfreq, yhmag, np.array([]), ystocEnv, Ns, H, fs)

# --------- write output sound ---------

UF.wavwrite(y,fs,'sax-phrase-short-synthesis.wav')
UF.wavwrite(yh,fs,'sax-phrase-harmonic-component.wav')
UF.wavwrite(yr,fs,'sax-phrase-residual-component.wav')
UF.wavwrite(yst,fs,'sax-phrase-stochastic-component.wav')
