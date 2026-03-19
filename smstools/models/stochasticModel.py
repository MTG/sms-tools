"""Stochastic model analysis and synthesis.

Implements frame-based stochastic envelope analysis and re-synthesis in linear or
mel frequency scale. Public API remains compatible with the original module.
"""

from typing import Optional

import numpy as np
from scipy.fft import fft, ifft
from scipy.interpolate import splev, splrep
from scipy.signal import resample
from scipy.signal.windows import hann

from smstools.models import utilFunctions as UF


def hertz_to_mel(f: float | np.ndarray) -> float | np.ndarray:
    """
    Convert Hertz to mel scale.

    Args:
        f: Frequency in Hertz (scalar or array).
    Returns:
        Frequency in mel scale (scalar or array).
    """
    return 2595 * np.log10(1 + f / 700)


def mel_to_hetz(m: float | np.ndarray) -> float | np.ndarray:
    """
    Convert mel scale to Hertz.

    Args:
        m: Frequency in mel scale (scalar or array).
    Returns:
        Frequency in Hertz (scalar or array).
    """
    return 700 * (10 ** (m / 2595) - 1)


def mel_to_hertz(m: float | np.ndarray) -> float | np.ndarray:
    """
    Convert mel scale to Hertz (spelling-correct alias).

    Args:
        m: Frequency in mel scale (scalar or array).
    Returns:
        Frequency in Hertz (scalar or array).
    """
    return mel_to_hetz(m)


def _validate_stochastic_params(H: int, N: int, stocf: float) -> None:
    """
    Validate common stochastic-model parameters.

    Ensures hop size and FFT size are valid and that the stochastic decimation
    factor is within supported limits.

    Args:
        H: Hop size.
        N: FFT size.
        stocf: Stochastic decimation factor.
    Raises:
        ValueError: If parameters are invalid.
    """
    hN = N // 2 + 1
    if hN * stocf < 3:
        raise ValueError("Stochastic decimation factor too small")
    if stocf > 1:
        raise ValueError("Stochastic decimation factor above 1")
    if H <= 0:
        raise ValueError("Hop size (H) smaller or equal to 0")
    if not UF.isPower2(N):
        raise ValueError("FFT size (N) is not a power of 2")


def _mel_grids(hN: int, fs: float, N: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Build mel-frequency grids used for interpolation.

    Returns both the original bin-frequency mel grid and a uniformly spaced mel
    grid of the same length for spline-based re-mapping.

    Args:
        hN: Number of positive FFT bins.
        fs: Sampling rate.
        N: FFT size.
    Returns:
        bin_freqs_mel: Mel scale of FFT bin frequencies.
        uniform_mel_freq: Uniformly spaced mel grid.
    """
    bin_freqs_mel = hertz_to_mel(np.arange(hN) * fs / float(N))
    uniform_mel_freq = np.linspace(bin_freqs_mel[0], bin_freqs_mel[-1], hN)
    return bin_freqs_mel, uniform_mel_freq


def _analyze_frame_to_env(
    mX: np.ndarray,
    stocf: float,
    mel_scale: bool,
    bin_freqs_mel: Optional[np.ndarray] = None,
    uniform_mel_freq: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Convert one magnitude spectrum frame to a decimated stochastic envelope.

    Optionally performs mel-domain interpolation before decimation.

    Args:
        mX: Magnitude spectrum frame.
        stocf: Decimation factor.
        mel_scale: Whether to use mel scale.
        bin_freqs_mel: Mel scale of FFT bin frequencies.
        uniform_mel_freq: Uniformly spaced mel grid.
    Returns:
        Decimated stochastic envelope.
    """
    mX_clip = np.maximum(-200, mX)
    env_size = int(stocf * mX.size)
    if mel_scale:
        spl = splrep(bin_freqs_mel, mX_clip)
        return resample(splev(uniform_mel_freq, spl), env_size)
    return resample(mX_clip, env_size)


def _synthesize_env_to_magnitude(
    env_frame: np.ndarray,
    hN: int,
    mel_scale: bool,
    bin_freqs_mel: np.ndarray | None = None,
    uniform_mel_freq: np.ndarray | None = None,
) -> np.ndarray:
    """
    Expand one stochastic envelope back to FFT-bin magnitude resolution.

    Optionally maps from uniform mel grid back to original FFT-bin mel spacing.

    Args:
        env_frame: Stochastic envelope frame.
        hN: Number of positive FFT bins.
        mel_scale: Whether to use mel scale.
        bin_freqs_mel: Mel scale of FFT bin frequencies.
        uniform_mel_freq: Uniformly spaced mel grid.
    Returns:
        Magnitude spectrum frame.
    """
    mY = resample(env_frame, hN)
    if mel_scale:
        spl = splrep(uniform_mel_freq, mY)
        return splev(bin_freqs_mel, spl)
    return mY


def _build_random_phase_spectrum(
    mY: np.ndarray, N: int, hN: int
) -> np.ndarray:
    """
    Create a Hermitian spectrum from magnitude with random positive-phase bins.

    The negative-frequency side is mirrored to ensure a real-valued inverse FFT.

    Args:
        mY: Magnitude spectrum frame.
        N: FFT size.
        hN: Number of positive FFT bins.
    Returns:
        Hermitian spectrum with random phase.
    """
    pY = 2 * np.pi * np.random.rand(hN)
    Y = np.zeros(N, dtype=complex)
    Y[:hN] = 10 ** (mY / 20) * np.exp(1j * pY)
    Y[hN:] = 10 ** (mY[-2:0:-1] / 20) * np.exp(-1j * pY[-2:0:-1])
    return Y


def stochasticModelAnal(
    x: np.ndarray,
    H: int,
    N: int,
    stocf: float,
    fs: float = 44100,
    melScale: int = 1,
) -> np.ndarray:
    """
    Analyze sound into stochastic spectral envelopes.

    Args:
        x: Input sound array.
        H: Hop size.
        N: FFT size.
        stocf: Decimation factor for stochastic envelope (0 < stocf <= 1).
        fs: Sampling rate.
        melScale: Use mel scale when 1, linear scale when 0.

    Returns:
        stocEnv: Stochastic envelopes, one row per frame.
    """

    _validate_stochastic_params(H, N, stocf)

    hN = N // 2 + 1
    no2 = N // 2
    x = np.concatenate([np.zeros(no2), x, np.zeros(no2)])
    pin = no2
    pend = x.size - no2
    mel_scale = melScale == 1
    if mel_scale:
        bin_freqs_mel, uniform_mel_freq = _mel_grids(hN, fs, N)
    else:
        bin_freqs_mel, uniform_mel_freq = None, None

    env_frames = []
    w = hann(N)
    while pin <= pend:
        xw = x[pin - no2 : pin + no2] * w
        X = fft(xw)
        mX = 20 * np.log10(np.abs(X[:hN]))
        mY = _analyze_frame_to_env(
            mX, stocf, mel_scale, bin_freqs_mel, uniform_mel_freq
        )
        env_frames.append(mY)
        pin += H

    return np.vstack(env_frames)


def stochasticModelSynth(
    stocEnv: np.ndarray, H: int, N: int, fs: float = 44100, melScale: int = 1
) -> np.ndarray:
    """
    Synthesize sound from stochastic envelopes.

    Args:
        stocEnv: Stochastic envelope, one row per frame.
        H: Hop size.
        N: FFT size.
        fs: Sampling rate.
        melScale: Use mel scale when 1, linear scale when 0.

    Returns:
        y: Output synthesized sound.
    """

    if not UF.isPower2(N):
        raise ValueError("N is not a power of two")

    hN = N // 2 + 1
    no2 = N // 2
    L = stocEnv.shape[0]
    ysize = H * (L + 3)
    y = np.zeros(ysize)
    ws = 2 * hann(N)
    pout = 0
    mel_scale = melScale == 1
    if mel_scale:
        bin_freqs_mel, uniform_mel_freq = _mel_grids(hN, fs, N)
    else:
        bin_freqs_mel, uniform_mel_freq = None, None

    for idx in range(L):
        mY = _synthesize_env_to_magnitude(
            stocEnv[idx, :], hN, mel_scale, bin_freqs_mel, uniform_mel_freq
        )
        Y = _build_random_phase_spectrum(mY, N, hN)
        fftbuffer = np.real(ifft(Y))
        y[pout : pout + N] += ws * fftbuffer
        pout += H

    y = y[no2 : y.size - no2]
    return y


def stochasticModel(
    x: np.ndarray,
    H: int,
    N: int,
    stocf: float,
    fs: float = 44100,
    melScale: int = 1,
) -> np.ndarray:
    """
    One-pass stochastic analysis/synthesis (frame-by-frame).

    Args:
        x: Input sound array.
        H: Hop size.
        N: FFT size.
        stocf: Decimation factor for stochastic envelope (0 < stocf <= 1).
        fs: Sampling rate.
        melScale: Use mel scale when 1, linear scale when 0.

    Returns:
        y: Output synthesized sound.
    """

    _validate_stochastic_params(H, N, stocf)

    # Analysis
    stocEnv = stochasticModelAnal(x, H, N, stocf, fs, melScale)
    # Synthesis
    y = stochasticModelSynth(stocEnv, H, N, fs, melScale)
    # Ensure output length matches input
    if len(y) > len(x):
        y = y[: len(x)]
    elif len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y
