"""Utility functions for spectral analysis, peak detection, and sinusoid synthesis.

Provides core signal processing operations including:
  - Audio I/O: wavread, wavwrite, wavplay
  - Spectral analysis: peakDetection, peakInterp with parabolic interpolation
  - Sinusoid modeling: genSpecSines (C-accelerated), sinewaveSynth, sineSubtraction
  - Fundamental frequency detection: f0Twm (Two-Way Mismatch algorithm)
  - Track cleaning and stochastic residual analysis
  - Helper functions: isPower2, sinc, genBhLobe

Uses Cython-accelerated C functions when available, falling back to pure-Python
implementations for portability.
"""

import copy
import os
import subprocess
import sys
import warnings

import numpy as np
from scipy.fft import fft, fftshift, ifft
from scipy.io.wavfile import read, write
from scipy.signal import resample
from scipy.signal.windows import blackmanharris, triang

try:
    from smstools.models.utilFunctions_C import utilFunctions_C as UF_C
except ImportError:
    UF_C = None
    warnings.warn(
        "Cython core functions were not imported. Falling back to pure-Python "
        "implementations; this may be slower. See README.md for build instructions.",
        RuntimeWarning,
    )

winsound_imported = False
if sys.platform == "win32":
    try:
        import winsound

        winsound_imported = True
    except ImportError:
        print("You won't be able to play sounds, winsound could not be imported")


def isPower2(num):
    """
    Check whether an integer is a power of two.
    """
    return ((num & (num - 1)) == 0) and num > 0


INT16_FAC = 2**15
INT32_FAC = 2**31
INT64_FAC = 2**63
norm_fact = {
    "int16": INT16_FAC,
    "int32": INT32_FAC,
    "int64": INT64_FAC,
    "float32": 1.0,
    "float64": 1.0,
}


def wavread(filename):
    """
    Read a mono audio file and return normalized floating-point samples.

    Args:
        filename: Path to input audio file.

    Returns:
        fs: Sampling rate of file.
        x: Floating-point mono signal in [-1, 1].

    Note:
        This function accepts any sampling rate and returns it as `fs`.
    """

    if not os.path.isfile(filename):
        raise ValueError("Input file is wrong")

    fs, x = read(filename)

    if len(x.shape) != 1:
        raise ValueError("Audio file should be mono")

    # Scale and convert audio to floating-point in [-1, 1].
    x = np.float32(x) / norm_fact[x.dtype.name]
    return fs, x


def wavplay(filename):
    """
    Play a WAV file using OS-specific system calls.

    Args:
        filename: Path to input audio file.
    """
    if not os.path.isfile(filename):
        print(
            "Input file does not exist. Make sure you computed the analysis/synthesis"
        )
    else:
        if sys.platform == "linux" or sys.platform == "linux2":
            # linux
            subprocess.call(["aplay", filename])

        elif sys.platform == "darwin":
            # OS X
            subprocess.call(["afplay", filename])
        elif sys.platform == "win32":
            if winsound_imported:
                winsound.PlaySound(filename, winsound.SND_FILENAME)
            else:
                print("Cannot play sound, winsound could not be imported")
        else:
            print("Platform not recognized")


def wavwrite(y, fs, filename):
    """
    Write a mono floating-point signal to a WAV file.

    Args:
        y: One-dimensional floating-point signal.
        fs: Sampling rate.
        filename: Output file path.
    """

    x = copy.deepcopy(y)
    x *= INT16_FAC
    x = np.int16(x)
    write(filename, fs, x)


def peakDetection(mX, t):
    """
    Detect spectral peak locations above threshold with local maxima constraint.

    Finds bins where magnitude is above threshold and higher than both neighbors,
    using efficient vectorized boolean operations.

    Args:
        mX: Magnitude spectrum (dB values)
        t: Detection threshold (dB)

    Returns:
        Indices of detected peaks (1-indexed in full spectrum)
    """
    # Vectorized boolean check: threshold + local maxima in one operation
    peaks = (mX[1:-1] > t) & (mX[1:-1] > mX[2:]) & (mX[1:-1] > mX[:-2])
    return np.where(peaks)[0] + 1  # Add 1 to compensate for [1:-1] slicing


def peakInterp(mX, pX, ploc):
    """
    Refine peak location and magnitude using parabolic interpolation.

    Fits a parabola through peak and its neighbors to estimate sub-bin peak
    location and magnitude. Uses linear interpolation for phase values.

    Args:
        mX: Magnitude spectrum (dB)
        pX: Phase spectrum (radians)
        ploc: Peak bin locations from peakDetection

    Returns:
        iploc: Interpolated peak locations (fractional bins)
        ipmag: Interpolated peak magnitudes
        ipphase: Interpolated peak phases
    """
    val = mX[ploc]  # Peak bin magnitude
    lval = mX[ploc - 1]  # Left neighbor magnitude
    rval = mX[ploc + 1]  # Right neighbor magnitude

    # Parabolic interpolation: solve for peak of parabola through 3 points
    denom = lval - 2 * val + rval
    # Avoid division by zero for numerically flat peaks
    denom = np.where(denom == 0, 1e-10, denom)

    iploc = ploc + 0.5 * (lval - rval) / denom  # Fractional peak location
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)  # Refined magnitude
    ipphase = np.interp(iploc, np.arange(pX.size), pX)  # Linear phase interpolation

    return iploc, ipmag, ipphase


def sinc(x, N):
    """
    Generate samples of the Dirichlet-kernel sinc main lobe.

    Args:
        x: Sample positions.
        N: FFT size used in the kernel expression.

    Returns:
        y: Sinc main-lobe values.
    """

    y = np.sin(N * x / 2) / np.sin(x / 2)
    y[np.isnan(y)] = N
    return y


def genBhLobe(x):
    """
    Generate the main lobe of a Blackman-Harris window spectrum.

    Args:
        x: Bin positions to compute (real-valued).

    Returns:
        y: Main-lobe spectrum values.
    """

    N = 512
    f = x * np.pi * 2 / N
    df = 2 * np.pi / N
    y = np.zeros(x.size)
    consts = [0.35875, 0.48829, 0.14128, 0.01168]
    for m in range(4):
        y += consts[m] / 2 * (sinc(f - df * m, N) + sinc(f + df * m, N))
    y = y / N / consts[0]
    return y


def genSpecSines(ipfreq, ipmag, ipphase, N, fs):
    """
    Generate sinusoidal spectrum, using C backend when available.

    Args:
        ipfreq, ipmag, ipphase: Peak frequencies, magnitudes, and phases.
        N: Complex spectrum size.
        fs: Sampling rate.

    Returns:
        Y: Generated complex sinusoidal spectrum.
    """

    if UF_C is not None:
        Y = UF_C.genSpecSines(N * ipfreq / float(fs), ipmag, ipphase, N)
    else:
        Y = genSpecSines_p(ipfreq, ipmag, ipphase, N, fs)
    return Y


def genSpecSines_p(ipfreq, ipmag, ipphase, N, fs):
    """
    Pure-Python sinusoidal spectrum generation.

    Args:
        ipfreq, ipmag, ipphase: Peak frequencies, magnitudes, and phases.
        N: Complex spectrum size.
        fs: Sampling rate.

    Returns:
        Y: Generated complex sinusoidal spectrum.
    """

    Y = np.zeros(N, dtype=complex)  # initialize output complex spectrum
    hN = N // 2  # size of positive freq. spectrum
    for i in range(0, ipfreq.size):  # generate all sine spectral lobes
        loc = N * ipfreq[i] / fs  # it should be in range ]0,hN-1[
        if loc == 0 or loc > hN - 1:
            continue
        binremainder = round(loc) - loc
        lb = np.arange(
            binremainder - 4, binremainder + 5
        )  # main lobe (real value) bins to read
        lmag = genBhLobe(lb) * 10 ** (
            ipmag[i] / 20
        )  # lobe magnitudes of the complex exponential
        b = np.arange(round(loc) - 4, round(loc) + 5, dtype="int")
        for m in range(0, 9):
            if b[m] < 0:  # peak lobe crosses DC bin
                Y[-b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] > hN:  # peak lobe croses Nyquist bin
                Y[2 * hN - b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] == 0 or b[m] == hN:  # peak lobe in the limits of the spectrum
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i]) + lmag[m] * np.exp(
                    -1j * ipphase[i]
                )
            else:  # peak lobe in positive freq. range
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i])
    Y[hN + 1 :] = Y[
        hN - 1 : 0 : -1
    ].conjugate()  # fill the negative part of the spectrum
    return Y


def sinewaveSynth(freqs, amp, H, fs):
    """
    Synthesize a sinusoid with time-varying frequency and amplitude.

    Generates smooth ramps for frequency and amplitude changes between frames,
    with automatic amplitude ramping at start/end to avoid clicks.

    Args:
        freqs: Array of frequencies (Hz), one per frame
        amp: Amplitude scaling factor
        H: Hop size (samples per frame)
        fs: Sampling rate (Hz)

    Returns:
        y: Output sinusoid samples
    """
    t = np.arange(H) / float(fs)  # Time array
    lastphase = 0  # Phase accumulation
    lastfreq = freqs[0]
    frames = []

    for l in range(freqs.size):
        if (lastfreq == 0) and (freqs[l] == 0):  # Silent to silent
            A = np.zeros(H)
            freq = np.zeros(H)
        elif (lastfreq == 0) and (freqs[l] > 0):  # Silent to tone (ramp up)
            A = np.arange(0, amp, amp / H)
            freq = np.ones(H) * freqs[l]
        elif (lastfreq > 0) and (freqs[l] > 0):  # Tone to tone
            A = np.ones(H) * amp
            if lastfreq == freqs[l]:
                freq = np.ones(H) * lastfreq
            else:
                freq = np.arange(lastfreq, freqs[l], (freqs[l] - lastfreq) / H)
        elif (lastfreq > 0) and (freqs[l] == 0):  # Tone to silent (ramp down)
            A = np.arange(amp, 0, -amp / H)
            freq = np.ones(H) * lastfreq

        # Generate phase and waveform
        phase = 2 * np.pi * freq * t + lastphase
        yh = A * np.cos(phase)
        lastfreq = freqs[l]
        lastphase = phase[H - 1] % (2 * np.pi)  # Wrap phase for continuity
        frames.append(yh)

    # Concatenate all frames efficiently
    return np.concatenate(frames)


def cleaningTrack(track, minTrackLength=3):
    """
    Remove short active fragments from a single track.

    Args:
        track: One-dimensional track array.
        minTrackLength: Minimum allowed fragment duration in frames.

    Returns:
        cleanTrack: Track with short active fragments zeroed out.
    """

    nFrames = track.size
    cleanTrack = np.copy(track)
    trackBegs = (
        np.nonzero(
            (track[: nFrames - 1] <= 0) & (track[1:] > 0)  # beginning of track contours
        )[0]
        + 1
    )
    if track[0] > 0:
        trackBegs = np.insert(trackBegs, 0, 0)
    trackEnds = np.nonzero((track[: nFrames - 1] > 0) & (track[1:] <= 0))[0] + 1
    if track[nFrames - 1] > 0:
        trackEnds = np.append(trackEnds, nFrames - 1)
    trackLengths = 1 + trackEnds - trackBegs
    for i, j in zip(trackBegs, trackLengths):
        if j <= minTrackLength:
            cleanTrack[i : i + j] = 0
    return cleanTrack


def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0, fs=None):
    """
    Wrapper around TWM f0 detection that selects candidate peaks and validates result.

    Args:
        pfreq: Peak frequencies (Hz)
        pmag: Peak magnitudes (dB)
        ef0max: Maximum accepted TWM error
        minf0: Minimum allowed f0 (Hz)
        maxf0: Maximum allowed f0 (Hz)
        f0t: Previous stable f0 estimate (Hz), 0 when unavailable
        fs: Optional sampling rate (Hz). If provided, maxf0 must be below Nyquist.

    Returns:
        f0: Estimated fundamental frequency in Hz, or 0 when rejected
    """
    if minf0 < 0:
        raise ValueError("Minimum fundamental frequency (minf0) smaller than 0")

    if maxf0 <= minf0:
        raise ValueError(
            "Maximum fundamental frequency (maxf0) must be bigger than minf0"
        )

    if (fs is not None) and (maxf0 >= fs / 2.0):
        raise ValueError(
            "Maximum fundamental frequency (maxf0) bigger than Nyquist frequency"
        )

    if (pfreq.size < 3) and (f0t == 0):
        return 0

    # Use only peaks within allowed f0 range.
    candidate_idx = np.flatnonzero((pfreq > minf0) & (pfreq < maxf0))
    if candidate_idx.size == 0:
        return 0
    f0_candidates = pfreq[candidate_idx]
    f0_candidate_mag = pmag[candidate_idx]

    # If previous f0 is available, prioritize candidates close to it.
    if f0t > 0:
        shortlist = np.flatnonzero(np.abs(f0_candidates - f0t) < f0t / 2.0)
        maxc = np.argmax(f0_candidate_mag)
        maxcfd = f0_candidates[maxc] % f0t
        if maxcfd > f0t / 2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd > (f0t / 4)):
            shortlist = np.append(maxc, shortlist)
        f0_candidates = f0_candidates[shortlist]

    if f0_candidates.size == 0:
        return 0

    if UF_C is not None:
        f0, f0error = UF_C.twm(pfreq, pmag, f0_candidates)
    else:
        f0, f0error = TWM_p(pfreq, pmag, f0_candidates)

    return f0 if (f0 > 0) and (f0error < ef0max) else 0


def TWM_p(pfreq, pmag, f0c):
    """
    Two-way mismatch algorithm for f0 detection (Beauchamp & Maher).

    Better performance is available via the C implementation: UF_C.twm.

    Args:
        pfreq: Peak frequencies (Hz)
        pmag: Peak magnitudes (dB)
        f0c: Candidate f0 values (Hz)

    Returns:
        f0: Detected fundamental frequency (Hz)
        f0Error: Error of selected candidate
    """

    p = 0.5
    q = 1.4
    r = 0.5
    rho = 0.33
    maxnpeaks = 10

    max_peak_mag = np.max(pmag)
    candidates = np.asarray(f0c, dtype=float)

    # Predicted-to-measured mismatch (PM).
    error_pm = np.zeros(candidates.size)
    n_pm = min(maxnpeaks, pfreq.size)
    harmonic = candidates.copy()
    for _ in range(n_pm):
        difmatrix_pm = np.abs(harmonic[:, None] - pfreq[None, :])
        freq_distance = np.amin(difmatrix_pm, axis=1)
        peak_loc = np.argmin(difmatrix_pm, axis=1)
        pondered_diff = freq_distance * (harmonic ** (-p))
        peak_mag = pmag[peak_loc]
        mag_factor = 10 ** ((peak_mag - max_peak_mag) / 20)
        error_pm += pondered_diff + mag_factor * (q * pondered_diff - r)
        harmonic += candidates

    # Measured-to-predicted mismatch (MP).
    error_mp = np.zeros(candidates.size)
    n_mp = min(maxnpeaks, pfreq.size)
    for i in range(candidates.size):
        nharm = np.round(pfreq[:n_mp] / candidates[i])
        nharm = (nharm >= 1) * nharm + (nharm < 1)
        freq_distance = np.abs(pfreq[:n_mp] - nharm * candidates[i])
        pondered_diff = freq_distance * (pfreq[:n_mp] ** (-p))
        peak_mag = pmag[:n_mp]
        mag_factor = 10 ** ((peak_mag - max_peak_mag) / 20)
        error_mp[i] = np.sum(
            mag_factor * (pondered_diff + mag_factor * (q * pondered_diff - r))
        )

    total_error = (error_pm / n_pm) + (rho * error_mp / n_mp)
    f0index = np.argmin(total_error)
    f0 = candidates[f0index]

    return f0, total_error[f0index]


def sineSubtraction(x, N, H, sfreq, smag, sphase, fs):
    """
    Subtract sinusoids from audio signal to extract residual.

    Removes spectral sines at given frequencies/magnitudes/phases,
    returning the residual sound component.

    Args:
        x: Input audio signal
        N: FFT size
        H: Hop size
        sfreq, smag, sphase: Sinusoid frequencies, magnitudes, phases (2D arrays: frames x sines)
        fs: Sampling rate

    Returns:
        xr: Residual signal after sine subtraction
    """
    hN = N // 2
    x = np.concatenate([np.zeros(hN), x, np.zeros(hN)])  # Pad for centered windowing
    bh = blackmanharris(N)
    w = bh / np.sum(bh)
    sw = np.zeros(N)
    sw[hN - H : hN + H] = triang(2 * H) / w[hN - H : hN + H]

    L = sfreq.shape[0]
    xr = np.zeros(x.size)
    pin = 0

    for l in range(L):
        xw = x[pin : pin + N] * w
        X = fft(fftshift(xw))
        if UF_C is not None:
            Yh = UF_C.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)
        else:
            Yh = genSpecSines_p(sfreq[l, :], smag[l, :], sphase[l, :], N, fs)
        Xr = X - Yh
        xrw = np.real(fftshift(ifft(Xr)))
        xr[pin : pin + N] += xrw * sw
        pin += H

    # Trim padding using slicing instead of delete
    xr = xr[hN : xr.size - hN]
    return xr


def stochasticResidualAnal(x, N, H, sfreq, smag, sphase, fs, stocf):
    """
    Subtract sinusoids and approximate the residual with a stochastic envelope.

    Args:
        x: Input sound
        N: FFT size
        H: Hop size
        sfreq, smag, sphase: Sinusoidal frequencies, magnitudes, and phases
        fs: Sampling rate
        stocf: Stochastic decimation factor

    Returns:
        stocEnv: Stochastic residual envelope per frame
    """

    hN = N // 2
    x = np.concatenate([np.zeros(hN), x, np.zeros(hN)])
    bh = blackmanharris(N)
    w = bh / np.sum(bh)
    L = sfreq.shape[0]
    env_size = int(hN * stocf)
    pin = 0
    env_frames = []

    for l in range(L):
        xw = x[pin : pin + N] * w
        X = fft(fftshift(xw))
        if UF_C is not None:
            Yh = UF_C.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)
        else:
            Yh = genSpecSines_p(sfreq[l, :], smag[l, :], sphase[l, :], N, fs)

        Xr = X - Yh
        mXr = 20 * np.log10(np.abs(Xr[:hN]))
        mXrenv = resample(np.maximum(-200, mXr), env_size)
        env_frames.append(mXrenv)
        pin += H

    if env_frames:
        stocEnv = np.vstack(env_frames)
    else:
        stocEnv = np.empty((0, env_size))

    return stocEnv
