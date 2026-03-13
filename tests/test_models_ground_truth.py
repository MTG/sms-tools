import numpy as np

from smstools.models import harmonicModel, hprModel, hpsModel, sprModel, spsModel, stochasticModel, utilFunctions
from smstools.models import dftModel as DFT
from smstools.models import sineModel as SM


FS = 44100


def _sine(freq, length, amp=0.8, phase=0.0, fs=FS):
    n = np.arange(length)
    return amp * np.sin(2 * np.pi * freq * n / fs + phase)


def _harmonic_stack(f0, length, harmonics=6, fs=FS):
    n = np.arange(length)
    x = np.zeros(length)
    for k in range(1, harmonics + 1):
        x += (1.0 / k) * np.sin(2 * np.pi * (k * f0) * n / fs)
    return x


def _snr_db(reference, estimate):
    error = reference - estimate
    num = np.sum(reference**2)
    den = np.sum(error**2) + np.finfo(float).eps
    return 10.0 * np.log10(num / den)


def test_single_sine_peak_frequency_accuracy():
    freq_true = 445.3
    M = 2047
    N = 8192
    x = _sine(freq_true, M)
    w = np.hanning(M)

    mX, pX = DFT.dftAnal(x, w, N)
    ploc = utilFunctions.peakDetection(mX, -120)
    iploc, ipmag, ipphase = utilFunctions.peakInterp(mX, pX, ploc)
    ipfreq = FS * iploc / float(N)

    freq_est = ipfreq[np.argmax(ipmag)]
    assert abs(freq_est - freq_true) < 3.0


def test_f0twm_recovers_ground_truth_from_harmonic_candidates():
    f0_true = 220.0
    pfreq = np.array([220.0, 440.0, 660.0, 880.0, 1100.0])
    pmag = np.array([0.0, -6.0, -9.5, -12.0, -14.0])

    f0_est = utilFunctions.f0Twm(
        pfreq=pfreq,
        pmag=pmag,
        ef0max=5,
        minf0=80,
        maxf0=500,
        f0t=0,
    )

    assert abs(f0_est - f0_true) < 1.0


def test_chirp_tracking_has_increasing_frequency_trend():
    length = 8192
    n = np.arange(length)
    f_start = 300.0
    f_end = 1200.0
    k = (f_end - f_start) / (length - 1)
    phase = 2 * np.pi * (f_start * n / FS + 0.5 * k * (n**2) / FS)
    x = 0.8 * np.sin(phase)

    tfreq, tmag, tphase = SM.sineModelAnal(
        x,
        fs=FS,
        w=np.hanning(1025),
        N=2048,
        H=128,
        t=-80,
        maxnSines=25,
        minSineDur=0.01,
    )

    frame_main_freq = []
    for frame in range(tfreq.shape[0]):
        valid = np.where(tfreq[frame] > 0)[0]
        if valid.size == 0:
            continue
        main_idx = valid[np.argmax(tmag[frame, valid])]
        frame_main_freq.append(tfreq[frame, main_idx])

    frame_main_freq = np.array(frame_main_freq)
    assert frame_main_freq.size > 5
    assert frame_main_freq[-1] > frame_main_freq[0]


def test_spr_component_additivity():
    x = _harmonic_stack(220.0, length=4096, harmonics=6) + 0.02 * np.random.default_rng(0).standard_normal(4096)
    w = np.hanning(513)

    y, ys, xr = sprModel.sprModel(x, fs=FS, w=w, N=1024, t=-80)

    assert y.shape == x.shape
    assert ys.shape == x.shape
    assert xr.shape == x.shape
    assert np.allclose(y, ys + xr, atol=1e-10)


def test_dft_roundtrip_meets_snr_threshold():
    M = 2047
    N = 8192
    x = _sine(440.0, length=M, amp=0.9) + 0.25 * _sine(880.0, length=M, amp=0.7)
    w = np.hanning(M)

    mX, pX = DFT.dftAnal(x, w, N)
    y = DFT.dftSynth(mX, pX, M)
    x_reference = x * (w / np.sum(w))

    assert y.shape == x.shape
    assert np.isfinite(y).all()
    assert _snr_db(x_reference, y) > 60.0


def test_harmonic_detection_recovers_expected_harmonics():
    f0 = 220.0
    pfreq = np.array([220.0, 440.0, 660.0, 880.0, 1000.0])
    pmag = np.array([-3.0, -6.0, -9.0, -12.0, -20.0])
    pphase = np.zeros_like(pfreq)

    hfreq, hmag, hphase = harmonicModel.harmonicDetection(
        pfreq=pfreq,
        pmag=pmag,
        pphase=pphase,
        f0=f0,
        nH=4,
        hfreqp=np.array([]),
        fs=FS,
    )

    assert np.allclose(hfreq, np.array([220.0, 440.0, 660.0, 880.0]), atol=1.0)
    assert hmag.shape == hfreq.shape
    assert hphase.shape == hfreq.shape


def test_stochastic_mel_hz_conversion_roundtrip():
    freqs = np.array([50.0, 220.0, 440.0, 1000.0, 5000.0])
    mels = stochasticModel.hertz_to_mel(freqs)
    recon = stochasticModel.mel_to_hetz(mels)

    assert np.allclose(freqs, recon, rtol=1e-8, atol=1e-8)


def test_stochastic_analysis_synthesis_produces_valid_signal():
    x = np.random.default_rng(42).standard_normal(4096)
    stoc_env = stochasticModel.stochasticModelAnal(x, H=128, N=512, stocf=0.5)
    y = stochasticModel.stochasticModelSynth(stoc_env, H=128, N=512)

    assert stoc_env.ndim == 2
    assert y.ndim == 1
    assert np.isfinite(stoc_env).all()
    assert np.isfinite(y).all()
    assert np.std(y) > 0


def test_hpr_component_additivity():
    x = _harmonic_stack(220.0, length=4096, harmonics=6)
    w = np.hanning(513)

    y, yh, xr = hprModel.hprModel(
        x,
        fs=FS,
        w=w,
        N=1024,
        t=-80,
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
    )

    assert y.shape == x.shape
    assert yh.shape == x.shape
    assert xr.shape == x.shape
    assert np.allclose(y, yh + xr, atol=1e-10)


def test_sps_component_additivity():
    x = _harmonic_stack(220.0, length=4096, harmonics=6)
    w = np.hanning(513)

    y, ys, yst = spsModel.spsModel(x, fs=FS, w=w, N=1024, t=-80, stocf=1)

    assert y.shape == x.shape
    assert ys.shape == x.shape
    assert yst.shape == x.shape
    assert np.allclose(y, ys + yst, atol=1e-10)


def test_hps_component_additivity():
    x = _harmonic_stack(220.0, length=4096, harmonics=6)
    w = np.hanning(513)

    y, yh, yst = hpsModel.hpsModel(
        x,
        fs=FS,
        w=w,
        N=1024,
        t=-80,
        nH=20,
        minf0=50,
        maxf0=500,
        f0et=5,
        stocf=1,
    )

    assert y.shape == x.shape
    assert yh.shape == x.shape
    assert yst.shape == x.shape
    assert np.allclose(y, yh + yst, atol=1e-10)
