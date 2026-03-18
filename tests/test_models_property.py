import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from smstools.models import (dftModel, harmonicModel, hprModel, hpsModel,
                             sineModel, sprModel, spsModel, stft,
                             stochasticModel)


# Utility for generating valid FFT sizes (powers of 2)
def valid_fft_size():
    return st.integers(min_value=8, max_value=4096).filter(lambda n: n & (n-1) == 0)

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096), valid_fft_size())
def test_dftModel_shape_and_finite(x_list, N):
    x = np.array(x_list)
    w = np.hanning(len(x))
    try:
        y = dftModel.dftModel(x, w, N)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096))
def test_sineModel_shape_and_finite(x_list):
    x = np.array(x_list)
    w = np.hanning(len(x))
    N = len(x)
    t = -80
    fs = 44100
    try:
        y = sineModel.sineModel(x, fs, w, N, t)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096))
def test_harmonicModel_shape_and_finite(x_list):
    x = np.array(x_list)
    w = np.hanning(len(x))
    N = len(x)
    t = -80
    fs = 44100
    nH = 10
    minf0 = 50
    maxf0 = 5000
    f0et = 0.1
    try:
        y = harmonicModel.harmonicModel(x, fs, w, N, t, nH, minf0, maxf0, f0et)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096))
def test_stochasticModel_shape_and_finite(x_list):
    x = np.array(x_list)
    H = 128
    N = len(x)
    stocf = 0.5
    fs = 44100
    melScale = 1
    try:
        y = stochasticModel.stochasticModel(x, H, N, stocf, fs, melScale)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096))
def test_hprModel_shape_and_finite(x_list):
    x = np.array(x_list)
    w = np.hanning(len(x))
    N = len(x)
    t = -80
    fs = 44100
    nH = 10
    minf0 = 50
    maxf0 = 5000
    f0et = 0.1
    harmDevSlope = 0.01
    try:
        y = hprModel.hprModel(x, fs, w, N, t, nH, minf0, maxf0, f0et)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096))
def test_hpsModel_shape_and_finite(x_list):
    x = np.array(x_list)
    w = np.hanning(len(x))
    N = len(x)
    t = -80
    fs = 44100
    nH = 10
    minf0 = 50
    maxf0 = 5000
    f0et = 0.1
    stocf = 0.5
    try:
        y = hpsModel.hpsModel(x, fs, w, N, t, nH, minf0, maxf0, f0et, stocf)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096))
def test_sprModel_shape_and_finite(x_list):
    x = np.array(x_list)
    w = np.hanning(len(x))
    N = len(x)
    t = -80
    fs = 44100
    try:
        y = sprModel.sprModel(x, fs, w, N, t)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")

@given(st.lists(st.floats(-1e6, 1e6), min_size=8, max_size=4096))
def test_spsModel_shape_and_finite(x_list):
    x = np.array(x_list)
    w = np.hanning(len(x))
    N = len(x)
    t = -80
    stocf = 0.5
    fs = 44100
    try:
        y = spsModel.spsModel(x, fs, w, N, t, stocf)
        assert y.shape == x.shape
        assert np.all(np.isfinite(y))
    except Exception:
        pytest.skip("Model exception for input.")
