"""
Tests comparing the Cython and pure-Python implementations of the two
functions that have both:

  • genSpecSines / genSpecSines_p
  • TWM  (UF_C.twm)  / TWM_p

Tests marked with ``cython_only`` are skipped automatically when the C
extension has not been compiled.  The remaining tests exercise the Python
paths so they always run.
"""

import numpy as np
import pytest

from smstools.models.utilFunctions import TWM_p, genSpecSines_p

# Try to import the C extension directly so we can call both sides.
# The test conftest always installs a stub into sys.modules, so we cannot use
# the 'is None' check.  Instead we probe: the stub's genSpecSines always returns
# zeros whilst the real Cython implementation returns non-zero values.
try:
    from smstools.models.utilFunctions_C import utilFunctions_C as _UF_C
except ImportError:
    _UF_C = None


def _is_real_cython() -> bool:
    """Return True only when the compiled C extension (not the test stub) is present."""
    if _UF_C is None:
        return False
    probe = _UF_C.genSpecSines(
        np.array([10.0]), np.array([0.0]), np.array([0.0]), 64
    )
    return bool(np.any(probe != 0))


cython_only = pytest.mark.skipif(
    not _is_real_cython(),
    reason="C extension not compiled (or only stub present) – skipping Cython-vs-Python comparison",
)

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

FS = 44100
N = 1024  # FFT size used throughout


def _sine_peaks(f0_hz, n_harmonics=6, fs=FS, n_fft=N):
    """Return (pfreq, pmag, pphase) arrays for a harmonic series."""
    freqs = np.array([f0_hz * k for k in range(1, n_harmonics + 1)], dtype=float)
    mags = np.array([-6.0 * k for k in range(n_harmonics)], dtype=float)
    phases = np.zeros(n_harmonics, dtype=float)
    return freqs, mags, phases


def _f0_candidates(f0_hz, n=5):
    """A small set of f0 candidates around the true f0."""
    return np.linspace(f0_hz * 0.8, f0_hz * 1.2, n)


# ---------------------------------------------------------------------------
# Python-path correctness tests (always run)
# ---------------------------------------------------------------------------


class TestGenSpecSinesPython:
    """Verify genSpecSines_p produces physically sensible output."""

    def test_output_shape_is_N(self):
        freqs, mags, phases = _sine_peaks(440.0)
        Y = genSpecSines_p(freqs, mags, phases, N, FS)
        assert Y.shape == (N,)

    def test_output_is_complex(self):
        freqs, mags, phases = _sine_peaks(440.0)
        Y = genSpecSines_p(freqs, mags, phases, N, FS)
        assert np.iscomplexobj(Y)

    def test_conjugate_symmetry(self):
        """Y[k] == conj(Y[N-k]) for k in 1..N//2-1 (Hermitian symmetry)."""
        freqs, mags, phases = _sine_peaks(440.0)
        Y = genSpecSines_p(freqs, mags, phases, N, FS)
        hN = N // 2
        np.testing.assert_allclose(
            Y[1:hN], Y[N - 1 : hN : -1].conj(), atol=1e-10,
            err_msg="genSpecSines_p output is not conjugate-symmetric",
        )

    def test_peak_bin_matches_frequency(self):
        """The bin with maximum magnitude should be close to the expected bin."""
        f0 = 440.0
        freqs = np.array([f0])
        mags = np.array([0.0])
        phases = np.array([0.0])
        Y = genSpecSines_p(freqs, mags, phases, N, FS)
        hN = N // 2
        peak_bin = np.argmax(np.abs(Y[:hN]))
        expected_bin = round(N * f0 / FS)
        assert abs(peak_bin - expected_bin) <= 1, (
            f"Peak at bin {peak_bin}, expected ~{expected_bin}"
        )

    def test_empty_input_returns_zeros(self):
        Y = genSpecSines_p(np.array([]), np.array([]), np.array([]), N, FS)
        np.testing.assert_array_equal(Y, np.zeros(N, dtype=complex))

    def test_out_of_range_frequencies_ignored(self):
        """Frequencies at DC or above Nyquist should produce near-zero output."""
        freqs = np.array([0.0, FS / 2.0 + 100.0])
        mags = np.array([0.0, 0.0])
        phases = np.array([0.0, 0.0])
        Y = genSpecSines_p(freqs, mags, phases, N, FS)
        assert np.max(np.abs(Y)) < 1e-10, "Out-of-range bins should be suppressed"

    def test_magnitude_scales_linearly(self):
        """Doubling the linear amplitude (+ 6 dB) should double the peak magnitude."""
        f0 = 880.0
        freqs = np.array([f0])
        phases = np.array([0.0])

        Y1 = genSpecSines_p(freqs, np.array([0.0]), phases, N, FS)
        Y2 = genSpecSines_p(freqs, np.array([6.0]), phases, N, FS)  # +6 dB ≈ ×2

        hN = N // 2
        peak1 = np.max(np.abs(Y1[:hN]))
        peak2 = np.max(np.abs(Y2[:hN]))
        np.testing.assert_allclose(peak2 / peak1, 2.0, rtol=0.02,
                                   err_msg="+6 dB should double linear amplitude")

    @pytest.mark.parametrize("fs", [22050, 44100, 48000])
    def test_conjugate_symmetry_at_multiple_rates(self, fs):
        freqs, mags, phases = _sine_peaks(440.0, fs=fs)
        # keep only harmonics below Nyquist
        mask = freqs < fs / 2 - 50
        Y = genSpecSines_p(freqs[mask], mags[mask], phases[mask], N, fs)
        hN = N // 2
        np.testing.assert_allclose(
            Y[1:hN], Y[N - 1 : hN : -1].conj(), atol=1e-10,
        )


class TestTWMPython:
    """Verify TWM_p produces physically sensible output."""

    def test_returns_float_tuple(self):
        freqs, mags, _ = _sine_peaks(220.0)
        f0c = _f0_candidates(220.0)
        f0, err = TWM_p(freqs, mags, f0c)
        assert isinstance(float(f0), float)
        assert isinstance(float(err), float)

    def test_detects_correct_f0(self):
        """TWM_p should pick the true f0 from a harmonic series."""
        f0_true = 220.0
        freqs, mags, _ = _sine_peaks(f0_true, n_harmonics=8)
        f0c = _f0_candidates(f0_true, n=9)
        f0, _ = TWM_p(freqs, mags, f0c)
        assert abs(f0 - f0_true) < f0_true * 0.15, (
            f"Expected f0≈{f0_true}, got {f0:.2f}"
        )

    def test_error_is_finite(self):
        """TWM_p error may be negative (due to the r-scaled magnitude term) but must be finite."""
        freqs, mags, _ = _sine_peaks(330.0)
        f0c = _f0_candidates(330.0)
        _, err = TWM_p(freqs, mags, f0c)
        assert np.isfinite(err)

    def test_correct_candidate_has_lowest_error(self):
        """The candidate closest to the true f0 should have the minimum error."""
        f0_true = 440.0
        freqs, mags, _ = _sine_peaks(f0_true, n_harmonics=8)
        candidates = np.array([f0_true * r for r in [0.5, 0.75, 1.0, 1.5, 2.0]])
        f0, _ = TWM_p(freqs, mags, candidates)
        assert abs(f0 - f0_true) / f0_true < 0.1, (
            f"Best candidate should be near {f0_true}, got {f0:.2f}"
        )

    def test_single_candidate(self):
        """TWM_p should work with a single f0 candidate."""
        f0_true = 110.0
        freqs, mags, _ = _sine_peaks(f0_true, n_harmonics=6)
        f0c = np.array([f0_true])
        f0, err = TWM_p(freqs, mags, f0c)
        assert f0 == pytest.approx(f0_true)

    @pytest.mark.parametrize("f0_true", [110.0, 220.0, 440.0, 880.0])
    def test_detects_various_f0s(self, f0_true):
        freqs, mags, _ = _sine_peaks(f0_true, n_harmonics=8)
        f0c = _f0_candidates(f0_true, n=11)
        f0, _ = TWM_p(freqs, mags, f0c)
        assert abs(f0 - f0_true) < f0_true * 0.15


# ---------------------------------------------------------------------------
# Cython-vs-Python comparison tests (skipped if C extension not built)
# ---------------------------------------------------------------------------


class TestGenSpecSinesCythonVsPython:
    """Compare UF_C.genSpecSines with genSpecSines_p output."""

    @cython_only
    def test_basic_agreement(self):
        freqs, mags, phases = _sine_peaks(440.0)
        bins = N * freqs / float(FS)
        Y_c = _UF_C.genSpecSines(bins, mags, phases, N)
        Y_p = genSpecSines_p(freqs, mags, phases, N, FS)
        mag_c = np.abs(Y_c)
        mag_p = np.abs(Y_p)
        significant = np.maximum(mag_c, mag_p) > 1e-3
        np.testing.assert_allclose(
            mag_c[significant], mag_p[significant], rtol=1.5e-2, atol=5e-4,
            err_msg="Magnitude spectra from C and Python differ on significant bins",
        )

    @cython_only
    def test_phase_agreement(self):
        freqs = np.array([440.0, 880.0])
        mags = np.array([0.0, -6.0])
        phases = np.array([0.3, -0.7])
        bins = N * freqs / float(FS)
        Y_c = _UF_C.genSpecSines(bins, mags, phases, N)
        Y_p = genSpecSines_p(freqs, mags, phases, N, FS)
        # Compare only the non-trivial positive-frequency bins
        hN = N // 2
        mag_c = np.abs(Y_c[1:hN])
        mag_p = np.abs(Y_p[1:hN])
        significant = np.maximum(mag_c, mag_p) > 1e-3
        phase_c = np.angle(Y_c[1:hN][significant])
        phase_p = np.angle(Y_p[1:hN][significant])
        phase_error = np.abs(np.angle(np.exp(1j * (phase_c - phase_p))))
        assert np.max(phase_error) < 0.08, "Phase spectra from C and Python differ"

    @cython_only
    @pytest.mark.parametrize("f0", [220.0, 440.0, 880.0])
    def test_peak_bin_agreement(self, f0):
        freqs = np.array([f0])
        mags = np.array([0.0])
        phases = np.array([0.0])
        bins = N * freqs / float(FS)
        Y_c = _UF_C.genSpecSines(bins, mags, phases, N)
        Y_p = genSpecSines_p(freqs, mags, phases, N, FS)
        hN = N // 2
        assert np.argmax(np.abs(Y_c[:hN])) == np.argmax(np.abs(Y_p[:hN])), (
            f"C and Python disagree on peak bin for f0={f0} Hz"
        )

    @cython_only
    def test_empty_input_agreement(self):
        Y_c = _UF_C.genSpecSines(np.array([]), np.array([]), np.array([]), N)
        Y_p = genSpecSines_p(np.array([]), np.array([]), np.array([]), N, FS)
        np.testing.assert_allclose(np.abs(Y_c), np.abs(Y_p), atol=1e-10)


class TestTWMCythonVsPython:
    """Compare UF_C.twm with TWM_p output."""

    @cython_only
    def test_f0_agreement(self):
        f0_true = 220.0
        freqs, mags, _ = _sine_peaks(f0_true, n_harmonics=8)
        f0c = _f0_candidates(f0_true, n=9)
        f0_c, _ = _UF_C.twm(freqs, mags, f0c)
        f0_p, _ = TWM_p(freqs, mags, f0c)
        assert abs(f0_c - f0_p) < 1.0, (
            f"C twm={f0_c:.2f} vs Python twm={f0_p:.2f} disagree"
        )

    @cython_only
    def test_error_agreement(self):
        f0_true = 440.0
        freqs, mags, _ = _sine_peaks(f0_true, n_harmonics=8)
        f0c = _f0_candidates(f0_true, n=9)
        f0_c, err_c = _UF_C.twm(freqs, mags, f0c)
        f0_p, err_p = TWM_p(freqs, mags, f0c)
        # Errors should be in the same ballpark (within 20 %)
        np.testing.assert_allclose(err_c, err_p, rtol=0.20,
                                   err_msg="C and Python TWM errors differ significantly")

    @cython_only
    @pytest.mark.parametrize("f0_true", [110.0, 220.0, 330.0, 440.0, 880.0])
    def test_same_winner_across_candidates(self, f0_true):
        """Both implementations should choose the same winning candidate."""
        freqs, mags, _ = _sine_peaks(f0_true, n_harmonics=8)
        f0c = _f0_candidates(f0_true, n=11)
        f0_c, _ = _UF_C.twm(freqs, mags, f0c)
        f0_p, _ = TWM_p(freqs, mags, f0c)
        # Both should pick the same element from f0c
        idx_c = np.argmin(np.abs(f0c - f0_c))
        idx_p = np.argmin(np.abs(f0c - f0_p))
        assert idx_c == idx_p, (
            f"f0_true={f0_true}: C picks candidate {idx_c} ({f0_c:.1f} Hz), "
            f"Python picks {idx_p} ({f0_p:.1f} Hz)"
        )
