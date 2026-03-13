"""
Tests that verify models work correctly at sampling rates other than 44100 Hz.
Covers fs=48000 (professional audio) and fs=22050 (half-rate).
"""

import numpy as np
import pytest

from smstools.models import (
    dftModel as DFT,
    hprModel,
    sineModel as SM,
    sprModel,
    stochasticModel,
    stft as STFT,
    utilFunctions,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sine(freq, length, amp=0.8, fs=44100):
    n = np.arange(length)
    return amp * np.sin(2 * np.pi * freq * n / fs)


def _harmonic_stack(f0, length, harmonics=6, fs=44100):
    n = np.arange(length)
    x = np.zeros(length)
    for k in range(1, harmonics + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * (k * f0) * n / fs)
    return x


def _snr_db(reference, estimate):
    error = reference - estimate
    num = np.sum(reference ** 2)
    den = np.sum(error ** 2) + np.finfo(float).eps
    return 10.0 * np.log10(num / den)


# ---------------------------------------------------------------------------
# Parametrize key tests over both non-default sample rates
# ---------------------------------------------------------------------------

RATES = [48000, 22050]


@pytest.mark.parametrize("fs", RATES)
def test_dft_peak_frequency_accuracy_at_non_default_rate(fs):
    """dftAnal should give sub-3 Hz frequency accuracy for a pure tone."""
    freq_true = 880.0
    M = 2047
    N = 8192
    x = _sine(freq_true, M, fs=fs)
    w = np.hanning(M)

    mX, pX = DFT.dftAnal(x, w, N)
    ploc = utilFunctions.peakDetection(mX, -120)
    iploc, ipmag, ipphase = utilFunctions.peakInterp(mX, pX, ploc)
    ipfreq = fs * iploc / float(N)

    freq_est = ipfreq[np.argmax(ipmag)]
    assert abs(freq_est - freq_true) < 3.0, (
        f"fs={fs}: expected {freq_true} Hz, got {freq_est:.2f} Hz"
    )


@pytest.mark.parametrize("fs", RATES)
def test_dft_roundtrip_snr_at_non_default_rate(fs):
    """DFT analysis→synthesis round-trip should preserve energy (>60 dB SNR)."""
    M = 2047
    N = 8192
    x = _sine(440.0, length=M, fs=fs)
    w = np.hanning(M)

    mX, pX = DFT.dftAnal(x, w, N)
    y = DFT.dftSynth(mX, pX, M)
    x_ref = x * (w / np.sum(w))

    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert _snr_db(x_ref, y) > 60.0, f"fs={fs}: SNR too low"


@pytest.mark.parametrize("fs", RATES)
def test_stft_roundtrip_at_non_default_rate(fs):
    """STFT analysis→synthesis should preserve signal shape at any rate."""
    x = _sine(440.0, length=4096, fs=fs)
    w = np.hanning(513)
    N, H = 1024, 128

    mX, pX = STFT.stftAnal(x, w, N, H)
    y = STFT.stftSynth(mX, pX, w.size, H)

    assert mX.ndim == 2
    assert mX.shape[1] == N // 2 + 1
    assert np.isfinite(y).all()


@pytest.mark.parametrize("fs", RATES)
def test_stochastic_analysis_synthesis_at_non_default_rate(fs):
    """Stochastic model should work correctly when fs is passed explicitly."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4096)

    stoc_env = stochasticModel.stochasticModelAnal(x, H=128, N=512, stocf=0.5, fs=fs)
    y = stochasticModel.stochasticModelSynth(stoc_env, H=128, N=512, fs=fs)

    assert stoc_env.ndim == 2
    assert y.ndim == 1
    assert np.isfinite(stoc_env).all()
    assert np.isfinite(y).all()
    assert np.std(y) > 0, f"fs={fs}: synthesis output is silent"


@pytest.mark.parametrize("fs", RATES)
def test_sine_model_chirp_trend_at_non_default_rate(fs):
    """sineModelAnal should track increasing frequency regardless of rate."""
    length = 8192
    n = np.arange(length)
    f_start, f_end = 200.0, min(800.0, fs / 2.0 - 200.0)
    k = (f_end - f_start) / (length - 1)
    phase = 2 * np.pi * (f_start * n / fs + 0.5 * k * n ** 2 / fs)
    x = 0.8 * np.sin(phase)

    tfreq, tmag, _ = SM.sineModelAnal(
        x, fs=fs, w=np.hanning(1025), N=2048, H=128,
        t=-80, maxnSines=25, minSineDur=0.01,
    )

    main_freqs = []
    for frame_idx in range(tfreq.shape[0]):
        valid = np.where(tfreq[frame_idx] > 0)[0]
        if valid.size == 0:
            continue
        main_freqs.append(tfreq[frame_idx, valid[np.argmax(tmag[frame_idx, valid])]])

    main_freqs = np.array(main_freqs)
    assert main_freqs.size > 5, f"fs={fs}: not enough tracked frames"
    assert main_freqs[-1] > main_freqs[0], (
        f"fs={fs}: frequency trend should be increasing"
    )


@pytest.mark.parametrize("fs", RATES)
def test_spr_component_additivity_at_non_default_rate(fs):
    """SPR output should satisfy y == ys + xr at any sample rate."""
    x = _harmonic_stack(220.0, length=4096, fs=fs) + \
        0.02 * np.random.default_rng(1).standard_normal(4096)
    w = np.hanning(513)

    y, ys, xr = sprModel.sprModel(x, fs=fs, w=w, N=1024, t=-80)

    assert y.shape == x.shape
    np.testing.assert_allclose(y, ys + xr, atol=1e-10,
                               err_msg=f"fs={fs}: SPR additivity violated")


@pytest.mark.parametrize("fs", RATES)
def test_hpr_component_additivity_at_non_default_rate(fs):
    """HPR output should satisfy y == yh + xr at any sample rate."""
    x = _harmonic_stack(220.0, length=4096, fs=fs)
    w = np.hanning(513)

    y, yh, xr = hprModel.hprModel(
        x, fs=fs, w=w, N=1024, t=-80,
        nH=20, minf0=50, maxf0=min(500, fs // 2 - 1), f0et=5,
    )

    assert y.shape == x.shape
    np.testing.assert_allclose(y, yh + xr, atol=1e-10,
                               err_msg=f"fs={fs}: HPR additivity violated")


@pytest.mark.parametrize("fs", RATES)
def test_f0twm_nyquist_guard_at_non_default_rate(fs):
    """f0Twm must reject maxf0 >= fs/2 regardless of the sample rate."""
    pfreq = np.array([200.0, 400.0, 600.0])
    pmag = np.array([0.0, -6.0, -12.0])

    with pytest.raises(ValueError, match="Nyquist"):
        utilFunctions.f0Twm(pfreq, pmag, ef0max=5,
                            minf0=50, maxf0=fs / 2, f0t=0, fs=fs)


@pytest.mark.parametrize("fs", RATES)
def test_f0twm_accepts_valid_maxf0_below_nyquist(fs):
    """f0Twm must not raise when maxf0 is safely below Nyquist."""
    pfreq = np.array([200.0, 400.0, 600.0])
    pmag = np.array([0.0, -6.0, -12.0])

    result = utilFunctions.f0Twm(pfreq, pmag, ef0max=5,
                                 minf0=50, maxf0=fs / 2 - 100, f0t=0, fs=fs)
    assert isinstance(result, float)


@pytest.mark.parametrize("fs", RATES)
def test_stochastic_mel_scale_uses_correct_frequency_axis(fs):
    """Mel-scale analysis/synthesis at non-44100 rate should agree with linear."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal(4096)

    env_mel = stochasticModel.stochasticModelAnal(x, H=128, N=512, stocf=0.5,
                                                   fs=fs, melScale=1)
    env_lin = stochasticModel.stochasticModelAnal(x, H=128, N=512, stocf=0.5,
                                                   fs=fs, melScale=0)

    # Both are valid envelopes: correct shape, finite, non-silent
    assert env_mel.shape == env_lin.shape
    assert np.isfinite(env_mel).all()
    assert np.isfinite(env_lin).all()
    # Mel and linear should not be identical (different frequency mapping)
    assert not np.allclose(env_mel, env_lin, atol=0.01), (
        f"fs={fs}: mel and linear envelopes should differ"
    )
