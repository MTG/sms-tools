# functions that implement analysis and synthesis of sounds using the Sinusoidal plus Residual Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal.windows import blackmanharris, triang
from scipy.fftpack import fft, ifft
import math
import dftModel as DFT
import sineModel as SM
import utilFunctions as UF
  
def sprModelAnal(x, fs, w, N, H, t, minSineDur, maxnSines, freqDevOffset, freqDevSlope):
    """
    Analysis of a sound using the sinusoidal plus residual model
    x: input sound, fs: sampling rate, w: analysis window; N: FFT size, t: threshold in negative dB,
    minSineDur: minimum duration of sinusoidal tracks
    maxnSines: maximum number of parallel sinusoids
    freqDevOffset: frequency deviation allowed in the sinusoids from frame to frame at frequency 0
    freqDevSlope: slope of the frequency deviation, higher frequencies have bigger deviation
    returns hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; xr: residual signal
    """

    # perform sinusoidal analysis
    tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
    Ns = 512
    xr = UF.sineSubtraction(x, Ns, H, tfreq, tmag, tphase, fs)    	# subtract sinusoids from original sound
    return tfreq, tmag, tphase, xr

def sprModelSynth(tfreq, tmag, tphase, xr, N, H, fs):
    """
    Synthesis of a sound using the sinusoidal plus residual model
    tfreq, tmag, tphase: sinusoidal frequencies, amplitudes and phases; stocEnv: stochastic envelope
    N: synthesis FFT size; H: hop size, fs: sampling rate
    returns y: output sound, y: sinusoidal component
    """

    ys = SM.sineModelSynth(tfreq, tmag, tphase, N, H, fs)          # synthesize sinusoids
    y = ys[:min(ys.size, xr.size)]+xr[:min(ys.size, xr.size)]   # sum sinusoids and residual components
    return y, ys

def sprModel(x, fs, w, N, t):
    """
    Analysis/synthesis of a sound using the sinusoidal plus residual model, one frame at a time
    x: input sound, fs: sampling rate, w: analysis window,
    N: FFT size (minimum 512), t: threshold in negative dB,
    returns y: output sound, ys: sinusoidal component, xr: residual component
    """

    hM1 = int(math.floor((w.size+1)/2))                           # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                               # half analysis window size by floor
    Ns = 512                                                      # FFT size for synthesis (even)
    H = Ns//4                                                     # Hop size used for analysis and synthesis
    hNs = Ns//2
    pin = max(hNs, hM1)                                           # initialize sound pointer in middle of analysis window
    pend = x.size - max(hNs, hM1)                                 # last sample to start a frame
    ysw = np.zeros(Ns)                                            # initialize output sound frame
    xrw = np.zeros(Ns)                                            # initialize output sound frame
    ys = np.zeros(x.size)                                         # initialize output array
    xr = np.zeros(x.size)                                         # initialize output array
    w = w / sum(w)                                                # normalize analysis window
    sw = np.zeros(Ns)
    ow = triang(2*H)                                              # overlapping window
    sw[hNs-H:hNs+H] = ow
    bh = blackmanharris(Ns)                                       # synthesis window
    bh = bh / sum(bh)                                             # normalize synthesis window
    wr = bh                                                       # window for residual
    sw[hNs-H:hNs+H] = sw[hNs-H:hNs+H] / bh[hNs-H:hNs+H]
    while pin<pend:
  #-----analysis-----             
        x1 = x[pin-hM1:pin+hM2]                                     # select frame
        mX, pX = DFT.dftAnal(x1, w, N)                              # compute dft
        ploc = UF.peakDetection(mX, t)                              # find peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)         # refine peak values		iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)          # refine peak values
        ipfreq = fs*iploc/float(N)                                  # convert peak locations to Hertz
        ri = pin-hNs-1                                              # input sound pointer for residual analysis
        xw2 = x[ri:ri+Ns]*wr                                        # window the input sound
        fftbuffer = np.zeros(Ns)                                    # reset buffer
        fftbuffer[:hNs] = xw2[hNs:]                                 # zero-phase window in fftbuffer
        fftbuffer[hNs:] = xw2[:hNs]
        X2 = fft(fftbuffer)                                         # compute FFT for residual analysis
  #-----synthesis-----
        Ys = UF.genSpecSines(ipfreq, ipmag, ipphase, Ns, fs)        # generate spec of sinusoidal component
        Xr = X2-Ys                                                  # get the residual complex spectrum
        fftbuffer = np.real(ifft(Ys))                               # inverse FFT of sinusoidal spectrum
        ysw[:hNs-1] = fftbuffer[hNs+1:]                             # undo zero-phase window
        ysw[hNs-1:] = fftbuffer[:hNs+1]
        fftbuffer = np.real(ifft(Xr))                               # inverse FFT of residual spectrum
        xrw[:hNs-1] = fftbuffer[hNs+1:]                             # undo zero-phase window
        xrw[hNs-1:] = fftbuffer[:hNs+1]
        ys[ri:ri+Ns] += sw*ysw                                      # overlap-add for sines
        xr[ri:ri+Ns] += sw*xrw                                      # overlap-add for residual
        pin += H                                                    # advance sound pointer
    y = ys+xr                                                       # sum of sinusoidal and residual components
    return y, ys, xr
