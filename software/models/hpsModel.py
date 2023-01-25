# functions that implement analysis and synthesis of sounds using the Harmonic plus Stochastic Model
# (for example usage check the examples models_interface)

import numpy as np
from scipy.signal import resample
from scipy.signal.windows import blackmanharris, triang, hann
from scipy.fft import fft, ifft
import math
import harmonicModel as HM
import sineModel as SM
import dftModel as DFT
import stochasticModel as STM
import utilFunctions as UF


def hpsModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur, Ns, stocf):
    """
	Analysis of a sound using the harmonic plus stochastic model
	x: input sound, fs: sampling rate, w: analysis window; N: FFT size, t: threshold in negative dB, 
	nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz, 
	maxf0: maximim f0 frequency in Hz; f0et: error threshold in the f0 detection (ex: 5),
	harmDevSlope: slope of harmonic deviation; minSineDur: minimum length of harmonics
	returns hfreq, hmag, hphase: harmonic frequencies, magnitude and phases; stocEnv: stochastic residual
	"""

    # perform harmonic analysis
    hfreq, hmag, hphase = HM.harmonicModelAnal(x, fs, w, N, H, t, nH, minf0, maxf0, f0et, harmDevSlope, minSineDur)
    # subtract sinusoids from original sound
    xr = UF.sineSubtraction(x, Ns, H, hfreq, hmag, hphase, fs)
    # perform stochastic analysis of residual
    stocEnv = STM.stochasticModelAnal(xr, H, H * 2, stocf)
    return hfreq, hmag, hphase, stocEnv


def hpsModelSynth(hfreq, hmag, hphase, stocEnv, N, H, fs):
    """
	Synthesis of a sound using the harmonic plus stochastic model
	hfreq, hmag: harmonic frequencies and amplitudes; stocEnv: stochastic envelope
	Ns: synthesis FFT size; H: hop size, fs: sampling rate 
	returns y: output sound, yh: harmonic component, yst: stochastic component
	"""

    yh = SM.sineModelSynth(hfreq, hmag, hphase, N, H, fs)  # synthesize harmonics
    yst = STM.stochasticModelSynth(stocEnv, H, H * 2)  # synthesize stochastic residual
    y = yh[:min(yh.size, yst.size)] + yst[:min(yh.size, yst.size)]  # sum harmonic and stochastic components
    return y, yh, yst


def hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, stocf):
    """
	Analysis/synthesis of a sound using the harmonic plus stochastic model, one frame at a time, no harmonic tracking
	x: input sound; fs: sampling rate, w: analysis window; N: FFT size (minimum 512), t: threshold in negative dB, 
	nH: maximum number of harmonics, minf0: minimum f0 frequency in Hz; maxf0: maximim f0 frequency in Hz, 
	f0et: error threshold in the f0 detection (ex: 5); stocf: decimation factor of mag spectrum for stochastic analysis
	returns y: output sound, yh: harmonic component, yst: stochastic component
	"""

    hM1 = int(math.floor((w.size + 1) / 2))  # half analysis window size by rounding
    hM2 = int(math.floor(w.size / 2))  # half analysis window size by floor
    Ns = 512  # FFT size for synthesis (even)
    H = Ns // 4  # Hop size used for analysis and synthesis
    hNs = Ns // 2
    pin = max(hNs, hM1)  # initialize sound pointer in middle of analysis window
    pend = x.size - max(hNs, hM1)  # last sample to start a frame
    yhw = np.zeros(Ns)  # initialize output sound frame
    ystw = np.zeros(Ns)  # initialize output sound frame
    yh = np.zeros(x.size)  # initialize output array
    yst = np.zeros(x.size)  # initialize output array
    w = w / sum(w)  # normalize analysis window
    sw = np.zeros(Ns)
    ow = triang(2 * H)  # overlapping window
    sw[hNs - H:hNs + H] = ow
    bh = blackmanharris(Ns)  # synthesis window
    bh = bh / sum(bh)  # normalize synthesis window
    wr = bh  # window for residual
    sw[hNs - H:hNs + H] = sw[hNs - H:hNs + H] / bh[hNs - H:hNs + H]  # synthesis window for harmonic component
    sws = H * hann(Ns) / 2  # synthesis window for stochastic
    hfreqp = []
    f0t = 0
    f0stable = 0
    while pin < pend:
        # -----analysis-----
        x1 = x[pin - hM1:pin + hM2]  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)  # compute dft
        ploc = UF.peakDetection(mX, t)  # find peaks
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)  # refine peak values
        ipfreq = fs * iploc / N  # convert peak locations to Hz
        f0t = UF.f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable == 0) & (f0t > 0)) \
                or ((f0stable > 0) & (np.abs(f0stable - f0t) < f0stable / 5.0)):
            f0stable = f0t  # consider a stable f0 if it is close to the previous one
        else:
            f0stable = 0
        hfreq, hmag, hphase = HM.harmonicDetection(ipfreq, ipmag, ipphase, f0t, nH, hfreqp, fs)  # find harmonics
        hfreqp = hfreq
        ri = pin - hNs - 1  # input sound pointer for residual analysis
        xw2 = x[ri:ri + Ns] * wr  # window the input sound
        fftbuffer = np.zeros(Ns)  # reset buffer
        fftbuffer[:hNs] = xw2[hNs:]  # zero-phase window in fftbuffer
        fftbuffer[hNs:] = xw2[:hNs]
        X2 = fft(fftbuffer)  # compute FFT for residual analysis
        # -----synthesis-----
        Yh = UF.genSpecSines(hfreq, hmag, hphase, Ns, fs)  # generate spec sines of harmonic component
        Xr = X2 - Yh  # get the residual complex spectrum
        mXr = 20 * np.log10(abs(Xr[:hNs]))  # magnitude spectrum of residual
        mXrenv = resample(np.maximum(-200, mXr),
                          mXr.size * stocf)  # decimate the magnitude spectrum and avoid -Inf
        stocEnv = resample(mXrenv, hNs)  # interpolate to original size
        pYst = 2 * np.pi * np.random.rand(hNs)  # generate phase random values
        Yst = np.zeros(Ns, dtype=complex)
        Yst[:hNs] = 10 ** (stocEnv / 20) * np.exp(1j * pYst)  # generate positive freq.
        Yst[hNs + 1:] = 10 ** (stocEnv[:0:-1] / 20) * np.exp(-1j * pYst[:0:-1])  # generate negative freq.

        fftbuffer = np.real(ifft(Yh))  # inverse FFT of harmonic spectrum
        yhw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
        yhw[hNs - 1:] = fftbuffer[:hNs + 1]

        fftbuffer = np.real(ifft(Yst))  # inverse FFT of stochastic spectrum
        ystw[:hNs - 1] = fftbuffer[hNs + 1:]  # undo zero-phase window
        ystw[hNs - 1:] = fftbuffer[:hNs + 1]

        yh[ri:ri + Ns] += sw * yhw  # overlap-add for sines
        yst[ri:ri + Ns] += sws * ystw  # overlap-add for stochastic
        pin += H  # advance sound pointer

    y = yh + yst  # sum of harmonic and stochastic components
    return y, yh, yst
