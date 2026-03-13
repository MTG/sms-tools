import numpy as np
import pytest
from scipy.io.wavfile import write

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
        match=r"Maximum fundamental frequency \(maxf0\) bigger than Nyquist frequency",
    ):
        utilFunctions.f0Twm(
            np.array([100.0, 200.0, 300.0]),
            np.array([0.0, -3.0, -6.0]),
            1.0,
            50.0,
            22050.0,
            fs=44100,
        )


def test_f0twm_rejects_maxf0_above_nyquist_at_48k():
    with pytest.raises(
        ValueError,
        match=r"Maximum fundamental frequency \(maxf0\) bigger than Nyquist frequency",
    ):
        utilFunctions.f0Twm(
            np.array([100.0, 200.0, 300.0]),
            np.array([0.0, -3.0, -6.0]),
            1.0,
            50.0,
            24000.0,
            fs=48000,
        )


def test_wavread_rejects_missing_file():
    with pytest.raises(ValueError, match="Input file is wrong"):
        utilFunctions.wavread("does_not_exist.wav")


def test_wavread_accepts_non_44100_sampling_rate(tmp_path):
    fs_in = 48000
    n = np.arange(1024)
    x = (0.2 * np.sin(2 * np.pi * 440.0 * n / fs_in) * 32767).astype(np.int16)
    wav_path = tmp_path / "tone_48k.wav"
    write(wav_path, fs_in, x)

    fs_out, y = utilFunctions.wavread(str(wav_path))

    assert fs_out == fs_in
    assert y.ndim == 1
    assert y.dtype == np.float32
