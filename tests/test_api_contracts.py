import inspect

import numpy as np

from smstools.models import dftModel, stft, utilFunctions


def test_dft_public_signatures_are_stable():
    assert str(inspect.signature(dftModel.dftModel)) == "(x, w, N)"
    assert str(inspect.signature(dftModel.dftAnal)) == "(x, w, N)"
    assert str(inspect.signature(dftModel.dftSynth)) == "(mX, pX, M)"


def test_stft_public_signatures_are_stable():
    assert str(inspect.signature(stft.stft)) == "(x, w, N, H)"
    assert str(inspect.signature(stft.stftAnal)) == "(x, w, N, H)"
    assert str(inspect.signature(stft.stftSynth)) == "(mY, pY, M, H)"


def test_utilfunctions_public_signatures_are_stable():
    assert str(inspect.signature(utilFunctions.isPower2)) == "(num)"
    assert str(inspect.signature(utilFunctions.peakDetection)) == "(mX, t)"
    assert str(inspect.signature(utilFunctions.peakInterp)) == "(mX, pX, ploc)"


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
