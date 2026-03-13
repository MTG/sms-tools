import numpy as np
import pytest

from smstools.models import dftModel, stft, utilFunctions


def test_dftanal_rejects_non_power_of_two_fft_size():
    x = np.ones(8)
    w = np.hanning(8)

    with pytest.raises(ValueError, match=r"FFT size \(N\) is not a power of 2"):
        dftModel.dftAnal(x, w, 1000)


def test_dftanal_rejects_window_larger_than_fft_size():
    x = np.ones(16)
    w = np.hanning(16)

    with pytest.raises(ValueError, match=r"Window size \(M\) is bigger than FFT size"):
        dftModel.dftAnal(x, w, 8)


def test_stft_rejects_non_positive_hop_size():
    x = np.ones(512)
    w = np.hanning(511)

    with pytest.raises(ValueError, match=r"Hop size \(H\) smaller or equal to 0"):
        stft.stft(x, w, 1024, 0)


def test_f0twm_rejects_negative_minf0():
    with pytest.raises(
        ValueError, match=r"Minimum fundamental frequency \(minf0\) smaller than 0"
    ):
        utilFunctions.f0Twm(np.array([100.0]), np.array([0.0]), 1.0, -1.0, 500.0)


def test_f0twm_rejects_too_large_maxf0():
    with pytest.raises(
        ValueError,
        match=r"Maximum fundamental frequency \(maxf0\) bigger than 10000Hz",
    ):
        utilFunctions.f0Twm(np.array([100.0]), np.array([0.0]), 1.0, 50.0, 10000.0)


def test_wavread_rejects_missing_file():
    with pytest.raises(ValueError, match="Input file is wrong"):
        utilFunctions.wavread("does_not_exist.wav")
