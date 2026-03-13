import inspect

import numpy as np
import pytest

from smstools.transformations import (
    harmonicTransformations,
    hpsTransformations,
    sineTransformations,
    stftTransformations,
    stochasticTransformations,
)


def _harmonic_tracks(frames=12, tracks=6):
    base = np.linspace(120.0, 1200.0, tracks)
    hfreq = np.tile(base, (frames, 1))
    hmag = np.tile(np.linspace(-20.0, -60.0, tracks), (frames, 1))
    return hfreq, hmag


def _stoc_env(frames=12, bins=64):
    env = np.linspace(-80.0, -20.0, bins)
    return np.tile(env, (frames, 1))


def _ensure_np_int_alias():
    if not hasattr(np, "int"):
        np.int = int


def test_transformations_public_signatures_are_stable():
    assert str(inspect.signature(sineTransformations.sineTimeScaling)) == "(sfreq, smag, timeScaling)"
    assert str(inspect.signature(sineTransformations.sineFreqScaling)) == "(sfreq, freqScaling)"

    assert str(inspect.signature(stochasticTransformations.stochasticTimeScale)) == "(stocEnv, timeScaling)"

    assert str(inspect.signature(harmonicTransformations.harmonicFreqScaling)) == (
        "(hfreq, hmag, freqScaling, freqStretching, timbrePreservation, fs)"
    )

    assert str(inspect.signature(hpsTransformations.hpsTimeScale)) == "(hfreq, hmag, stocEnv, timeScaling)"
    assert str(inspect.signature(hpsTransformations.hpsMorph)) == (
        "(hfreq1, hmag1, stocEnv1, hfreq2, hmag2, stocEnv2, hfreqIntp, hmagIntp, stocIntp)"
    )

    assert str(inspect.signature(stftTransformations.stftFiltering)) == "(x, fs, w, N, H, filter)"
    assert str(inspect.signature(stftTransformations.stftMorph)) == (
        "(x1, x2, fs, w1, N1, w2, N2, H1, smoothf, balancef)"
    )


def test_sine_transformations_smoke():
    sfreq, smag = _harmonic_tracks(frames=10, tracks=5)

    ysfreq, ysmag = sineTransformations.sineTimeScaling(
        sfreq, smag, np.array([0.0, 0.0, 1.0, 1.0])
    )
    yfreq = sineTransformations.sineFreqScaling(sfreq, np.array([0.0, 1.0, 1.0, 1.0]))

    assert ysfreq.shape == sfreq.shape
    assert ysmag.shape == smag.shape
    assert yfreq.shape == sfreq.shape
    assert np.isfinite(ysfreq).all()
    assert np.isfinite(ysmag).all()
    assert np.isfinite(yfreq).all()


def test_stochastic_transformation_smoke():
    stoc = _stoc_env(frames=10, bins=32)

    ystoc = stochasticTransformations.stochasticTimeScale(
        stoc, np.array([0.0, 0.0, 1.0, 1.0])
    )

    assert ystoc.shape == stoc.shape
    assert np.isfinite(ystoc).all()


def test_harmonic_transformation_smoke():
    hfreq, hmag = _harmonic_tracks(frames=10, tracks=6)

    yhfreq, yhmag = harmonicTransformations.harmonicFreqScaling(
        hfreq,
        hmag,
        np.array([0.0, 1.0, 1.0, 1.0]),
        np.array([0.0, 1.0, 1.0, 1.0]),
        timbrePreservation=0,
        fs=44100,
    )

    assert yhfreq.shape == hfreq.shape
    assert yhmag.shape == hmag.shape
    assert np.isfinite(yhfreq).all()
    assert np.isfinite(yhmag).all()


def test_hps_transformations_smoke():
    _ensure_np_int_alias()
    hfreq1, hmag1 = _harmonic_tracks(frames=10, tracks=6)
    hfreq2, hmag2 = _harmonic_tracks(frames=8, tracks=6)
    stoc1 = _stoc_env(frames=10, bins=24)
    stoc2 = _stoc_env(frames=8, bins=24)

    yhfreq, yhmag, ystoc = hpsTransformations.hpsTimeScale(
        hfreq1,
        hmag1,
        stoc1,
        np.array([0.0, 0.0, 1.0, 1.0]),
    )

    mhfreq, mhmag, mstoc = hpsTransformations.hpsMorph(
        hfreq1,
        hmag1,
        stoc1,
        hfreq2,
        hmag2,
        stoc2,
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        np.array([0.0, 0.0, 1.0, 1.0]),
    )

    assert yhfreq.shape == hfreq1.shape
    assert yhmag.shape == hmag1.shape
    assert ystoc.shape == stoc1.shape

    assert mhfreq.shape == hfreq1.shape
    assert mhmag.shape == hmag1.shape
    assert mstoc.shape == stoc1.shape


def test_stft_transformations_smoke():
    fs = 44100
    x1 = np.random.default_rng(0).standard_normal(4096)
    x2 = np.random.default_rng(1).standard_normal(4096)
    w = np.hanning(511)
    N = 1024
    H = 128
    filt = np.zeros((N // 2) + 1)

    yf = stftTransformations.stftFiltering(x1, fs, w, N, H, filt)
    ym = stftTransformations.stftMorph(x1, x2, fs, w, N, w, N, H, smoothf=1.0, balancef=0.5)

    assert yf.shape == x1.shape
    assert ym.shape == x1.shape
    assert np.isfinite(yf).all()
    assert np.isfinite(ym).all()


def test_transformation_error_contracts():
    sfreq, smag = _harmonic_tracks(frames=4, tracks=3)
    hfreq, hmag = _harmonic_tracks(frames=4, tracks=3)
    stoc = _stoc_env(frames=4, bins=8)

    with pytest.raises(ValueError, match="even size"):
        sineTransformations.sineTimeScaling(sfreq, smag, np.array([0.0, 1.0, 2.0]))

    with pytest.raises(ValueError, match="even size"):
        harmonicTransformations.harmonicFreqScaling(
            hfreq,
            hmag,
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 1.0, 1.0]),
            timbrePreservation=0,
            fs=44100,
        )

    with pytest.raises(ValueError, match="even size"):
        stochasticTransformations.stochasticTimeScale(stoc, np.array([0.0, 1.0, 2.0]))

    x = np.random.default_rng(2).standard_normal(2048)
    w = np.hanning(511)

    with pytest.raises(ValueError, match="Smooth factor too small"):
        stftTransformations.stftMorph(x, x, 44100, w, 1024, w, 1024, 128, smoothf=0.001, balancef=0.5)

    with pytest.raises(ValueError, match="Balance factor outside range"):
        stftTransformations.stftMorph(x, x, 44100, w, 1024, w, 1024, 128, smoothf=1.0, balancef=1.5)
