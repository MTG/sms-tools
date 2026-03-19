import inspect

import numpy as np

from smstools.models import dftModel, stft, utilFunctions


def test_dft_public_signatures_are_stable():
    assert (
        str(inspect.signature(dftModel.dftModel))
        == "(x: numpy.ndarray, w: numpy.ndarray, N: int) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(dftModel.dftAnal))
        == "(x: numpy.ndarray, w: numpy.ndarray, N: int) -> tuple[numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(dftModel.dftSynth))
        == "(mX: numpy.ndarray, pX: numpy.ndarray, M: int) -> numpy.ndarray"
    )


def test_stft_public_signatures_are_stable():
    assert (
        str(inspect.signature(stft.stft))
        == "(x: numpy.ndarray, w: numpy.ndarray, N: int, H: int) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(stft.stftAnal))
        == "(x: numpy.ndarray, w: numpy.ndarray, N: int, H: int) -> tuple[numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(stft.stftSynth))
        == "(mY: numpy.ndarray, pY: numpy.ndarray, M: int, H: int) -> numpy.ndarray"
    )


def test_utilfunctions_public_signatures_are_stable():
    assert (
        str(inspect.signature(utilFunctions.isPower2)) == "(num: int) -> bool"
    )
    assert (
        str(inspect.signature(utilFunctions.peakDetection))
        == "(mX: numpy.ndarray, t: float) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(utilFunctions.peakInterp))
        == "(mX: numpy.ndarray, pX: numpy.ndarray, ploc: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )


def test_dftanal_return_format_is_two_1d_arrays():
    M = 511
    N = 1024
    x = np.random.default_rng(0).standard_normal(M)
    w = np.hanning(M)

    mX, pX = dftModel.dftAnal(x, w, N)

    assert isinstance(mX, np.ndarray)
    assert isinstance(pX, np.ndarray)
    assert mX.ndim == 1
    assert pX.ndim == 1
    assert mX.shape == pX.shape
    assert mX.size == (N // 2) + 1


def test_stftanal_return_format_is_two_2d_arrays():
    M = 511
    N = 1024
    H = 128
    x = np.random.default_rng(1).standard_normal(4096)
    w = np.hanning(M)

    mY, pY = stft.stftAnal(x, w, N, H)

    assert isinstance(mY, np.ndarray)
    assert isinstance(pY, np.ndarray)
    assert mY.ndim == 2
    assert pY.ndim == 2
    assert mY.shape == pY.shape
    assert mY.shape[1] == (N // 2) + 1
