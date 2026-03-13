import numpy as np

from smstools.models import dftModel, stft


def test_dftmodel_zero_input_returns_zero_with_same_shape():
    M = 511
    N = 1024
    x = np.zeros(M)
    w = np.hanning(M)

    y = dftModel.dftModel(x, w, N)

    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.all(y == 0)


def test_dftanal_and_dftsynth_produce_expected_length_and_finite_values():
    M = 511
    N = 1024
    t = np.arange(M) / 44100.0
    x = np.sin(2 * np.pi * 440.0 * t)
    w = np.hanning(M)

    mX, pX = dftModel.dftAnal(x, w, N)
    y = dftModel.dftSynth(mX, pX, M)

    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_stft_roundtrip_preserves_length_and_is_finite():
    M = 511
    N = 1024
    H = 128
    x = np.random.default_rng(2).standard_normal(4096)
    w = np.hanning(M)

    y = stft.stft(x, w, N, H)

    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert np.any(np.abs(y) > 0)
