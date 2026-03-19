import inspect

import numpy as np

from smstools.models import (
    dftModel,
    harmonicModel,
    hprModel,
    hpsModel,
    sineModel,
    sprModel,
    spsModel,
    stft,
    stochasticModel,
)


def _signal(fs=44100, length=4096):
    n = np.arange(length)
    x = 0.8 * np.sin(2 * np.pi * 220.0 * n / fs)
    x += 0.2 * np.sin(2 * np.pi * 440.0 * n / fs)
    return x.astype(float)


def _analysis_params():
    return {
        "fs": 44100,
        "w": np.hanning(513),
        "N": 1024,
        "H": 128,
        "t": -80,
    }


def test_model_module_signatures_are_stable():
    # Updated signature to match new API
    assert (
        str(inspect.signature(sineModel.sineModel))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, t: float, max_n_sines: int = 100, min_sine_dur: float = 0.01, freq_dev_offset: float = 20, freq_dev_slope: float = 0.01) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(sineModel.sineModelAnal))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, maxnSines: int = 100, minSineDur: float = 0.01, freqDevOffset: float = 20, freqDevSlope: float = 0.01) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(sineModel.sineModelSynth))
        == "(tfreq: numpy.ndarray, tmag: numpy.ndarray, tphase: numpy.ndarray, N: int, H: int, fs: float) -> numpy.ndarray"
    )

    assert (
        str(inspect.signature(harmonicModel.f0Detection))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, minf0: float, maxf0: float, f0et: float) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(harmonicModel.harmonicModel))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, t: float, nH: int, minf0: float, maxf0: float, f0et: float) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(harmonicModel.harmonicModelAnal))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, nH: int, minf0: float, maxf0: float, f0et: float, harmDevSlope: float = 0.01, minSineDur: float = 0.02) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )

    assert (
        str(inspect.signature(stochasticModel.stochasticModelAnal))
        == "(x: numpy.ndarray, H: int, N: int, stocf: float, fs: float = 44100, melScale: int = 1) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(stochasticModel.stochasticModelSynth))
        == "(stocEnv: numpy.ndarray, H: int, N: int, fs: float = 44100, melScale: int = 1) -> numpy.ndarray"
    )
    assert (
        str(inspect.signature(stochasticModel.stochasticModel))
        == "(x: numpy.ndarray, H: int, N: int, stocf: float, fs: float = 44100, melScale: int = 1) -> numpy.ndarray"
    )

    assert (
        str(inspect.signature(sprModel.sprModelAnal))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, minSineDur: float, maxnSines: int, freqDevOffset: float, freqDevSlope: float) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(sprModel.sprModelSynth))
        == "(tfreq: numpy.ndarray, tmag: numpy.ndarray, tphase: numpy.ndarray, xr: numpy.ndarray, N: int, H: int, fs: float) -> tuple[numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(sprModel.sprModel))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, minSineDur: float = 0.02, maxnSines: int = 100, freqDevOffset: float = 20, freqDevSlope: float = 0.01) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )

    assert (
        str(inspect.signature(spsModel.spsModelAnal))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, minSineDur: float, maxnSines: int, freqDevOffset: float, freqDevSlope: float, stocf: float) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(spsModel.spsModelSynth))
        == "(tfreq: numpy.ndarray, tmag: numpy.ndarray, tphase: numpy.ndarray, stocEnv: numpy.ndarray, N: int, H: int, fs: float) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(spsModel.spsModel))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int = 128, t: float = -80, minSineDur: float = 0.02, maxnSines: int = 100, freqDevOffset: float = 20.0, freqDevSlope: float = 0.01, Ns: int = 512, stocf: float = 0.2) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )

    assert (
        str(inspect.signature(hprModel.hprModelAnal))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, minSineDur: float, nH: int, minf0: float, maxf0: float, f0et: float, harmDevSlope: float) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(hprModel.hprModelSynth))
        == "(hfreq: numpy.ndarray, hmag: numpy.ndarray, hphase: numpy.ndarray, xr: numpy.ndarray, N: int, H: int, fs: float) -> Tuple[numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(hprModel.hprModel))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int = 128, t: float = -80, nH: int = 100, minf0: float = 50.0, maxf0: float = 5000.0, f0et: float = 5.0, harmDevSlope: float = 0.01, minSineDur: float = 0.02, Ns: int = 512) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )

    assert (
        str(inspect.signature(hpsModel.hpsModelAnal))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int, t: float, nH: int, minf0: float, maxf0: float, f0et: float, harmDevSlope: float, minSineDur: float, Ns: int, stocf: float) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(hpsModel.hpsModelSynth))
        == "(hfreq: numpy.ndarray, hmag: numpy.ndarray, hphase: numpy.ndarray, stocEnv: numpy.ndarray, N: int, H: int, fs: float) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )
    assert (
        str(inspect.signature(hpsModel.hpsModel))
        == "(x: numpy.ndarray, fs: float, w: numpy.ndarray, N: int, H: int = 128, t: float = -80, nH: int = 100, minf0: float = 50.0, maxf0: float = 5000.0, f0et: float = 5.0, harmDevSlope: float = 0.01, minSineDur: float = 0.02, Ns: int = 512, stocf: float = 0.2) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]"
    )


def test_dftmodel_zero_input_returns_zero_with_same_shape():
    M = 511
    N = 1024
    x = np.zeros(M)
    w = np.hanning(M)

    y = dftModel.dftModel(x, w, N)

    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.allclose(y, 0, atol=1e-12)


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


def test_sine_model_analysis_and_synthesis_shapes_are_stable():
    params = _analysis_params()
    x = _signal()

    tfreq, tmag, tphase = sineModel.sineModelAnal(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        maxnSines=20,
    )
    y = sineModel.sineModelSynth(
        tfreq, tmag, tphase, 512, params["H"], params["fs"]
    )

    assert tfreq.ndim == 2
    assert tmag.shape == tfreq.shape
    assert tphase.shape == tfreq.shape
    assert y.ndim == 1
    assert np.isfinite(y).all()


def test_harmonic_model_analysis_and_synthesis_shapes_are_stable():
    params = _analysis_params()
    x = _signal()

    xhfreq, xhmag, xhphase = harmonicModel.harmonicModelAnal(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
    )
    y = harmonicModel.harmonicModel(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["t"],
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
    )

    assert xhfreq.ndim == 2
    assert xhmag.shape == xhfreq.shape
    assert xhphase.shape == xhfreq.shape
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_stochastic_model_analysis_and_synthesis_shapes_are_stable():
    x = _signal()
    H = 128
    N = 512

    stocEnv = stochasticModel.stochasticModelAnal(x, H=H, N=N, stocf=0.5)
    y_synth = stochasticModel.stochasticModelSynth(stocEnv, H=H, N=N)
    y = stochasticModel.stochasticModel(x, H=H, N=N, stocf=0.5)

    assert stocEnv.ndim == 2
    assert y_synth.ndim == 1
    assert y.shape == x.shape
    assert np.isfinite(stocEnv).all()
    assert np.isfinite(y_synth).all()
    assert np.isfinite(y).all()


def test_spr_model_analysis_synthesis_and_full_model_shapes_are_stable():
    params = _analysis_params()
    x = _signal()

    tfreq, tmag, tphase, xr = sprModel.sprModelAnal(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        minSineDur=0.01,
        maxnSines=20,
        freqDevOffset=20,
        freqDevSlope=0.01,
    )
    y_synth, ys = sprModel.sprModelSynth(
        tfreq, tmag, tphase, xr, 512, params["H"], params["fs"]
    )
    y, ys_full, xr_full = sprModel.sprModel(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        minSineDur=0.01,
        maxnSines=20,
        freqDevOffset=20,
        freqDevSlope=0.01,
    )

    assert tfreq.ndim == 2
    assert tmag.shape == tfreq.shape
    assert tphase.shape == tfreq.shape
    assert xr.ndim == 1
    assert y_synth.ndim == 1
    assert ys.ndim == 1
    assert y.shape == x.shape
    assert ys_full.shape == x.shape
    assert xr_full.shape == x.shape


def test_sps_model_analysis_synthesis_and_full_model_shapes_are_stable():
    params = _analysis_params()
    x = _signal()

    tfreq, tmag, tphase, stocEnv = spsModel.spsModelAnal(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        minSineDur=0.01,
        maxnSines=20,
        freqDevOffset=20,
        freqDevSlope=0.01,
        stocf=1,
    )
    y_synth, ys, yst = spsModel.spsModelSynth(
        tfreq, tmag, tphase, stocEnv, 512, params["H"], params["fs"]
    )
    y, ys_full, yst_full = spsModel.spsModel(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        minSineDur=0.01,
        maxnSines=20,
        freqDevOffset=20,
        freqDevSlope=0.01,
        stocf=1,
    )

    assert tfreq.ndim == 2
    assert tmag.shape == tfreq.shape
    assert tphase.shape == tfreq.shape
    assert stocEnv.ndim == 2
    assert y_synth.ndim == 1
    assert ys.ndim == 1
    assert yst.ndim == 1
    assert y.shape == x.shape
    assert ys_full.shape == x.shape
    assert yst_full.shape == x.shape


def test_hpr_model_analysis_synthesis_and_full_model_shapes_are_stable():
    params = _analysis_params()
    x = _signal()

    hfreq, hmag, hphase, xr = hprModel.hprModelAnal(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        minSineDur=0.01,
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
        harmDevSlope=0.01,
    )
    y_synth, yh = hprModel.hprModelSynth(
        hfreq, hmag, hphase, xr, 512, params["H"], params["fs"]
    )
    y, yh_full, xr_full = hprModel.hprModel(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        minSineDur=0.01,
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
        harmDevSlope=0.01,
    )

    assert hfreq.ndim == 2
    assert hmag.shape == hfreq.shape
    assert hphase.shape == hfreq.shape
    assert xr.ndim == 1
    assert y_synth.ndim == 1
    assert yh.ndim == 1
    assert y.shape == x.shape
    assert yh_full.shape == x.shape
    assert xr_full.shape == x.shape


def test_hps_model_analysis_synthesis_and_full_model_shapes_are_stable():
    params = _analysis_params()
    x = _signal()

    hfreq, hmag, hphase, stocEnv = hpsModel.hpsModelAnal(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
        harmDevSlope=0.01,
        minSineDur=0.01,
        Ns=512,
        stocf=1,
    )
    y_synth, yh, yst = hpsModel.hpsModelSynth(
        hfreq, hmag, hphase, stocEnv, 512, params["H"], params["fs"]
    )
    y, yh_full, yst_full = hpsModel.hpsModel(
        x,
        params["fs"],
        params["w"],
        params["N"],
        params["H"],
        params["t"],
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
        stocf=1,
    )

    assert hfreq.ndim == 2
    assert hmag.shape == hfreq.shape
    assert hphase.shape == hfreq.shape
    assert stocEnv.ndim == 2
    assert y_synth.ndim == 1
    assert yh.ndim == 1
    assert yst.ndim == 1
    assert y.shape == x.shape
    assert yh_full.shape == x.shape
    assert yst_full.shape == x.shape
