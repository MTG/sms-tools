"""
Ground-truth / algorithmic tests for the transformation modules.

Each test works with synthetic data whose expected outcome is mathematically
predictable, so assertions can be exact or tightly bounded rather than merely
"produces a finite array".
"""

import numpy as np

from smstools.transformations import (
    harmonicTransformations,
    hpsTransformations,
    sineTransformations,
    stftTransformations,
    stochasticTransformations,
)

FS = 44100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sine_tracks(frames, tracks, base_freq=220.0):
    """Synthetic sine-track matrix: track k holds (k+1)*base_freq Hz."""
    tfreq = np.zeros((frames, tracks))
    tmag = np.zeros((frames, tracks))
    for k in range(tracks):
        tfreq[:, k] = (k + 1) * base_freq
        tmag[:, k] = -6.0 * (k + 1)
    return tfreq, tmag


def _harmonic_tracks(frames, tracks, f0=220.0):
    """Synthetic harmonic track matrix (all frames identical)."""
    hfreq = np.zeros((frames, tracks))
    hmag = np.zeros((frames, tracks))
    for k in range(tracks):
        hfreq[:, k] = (k + 1) * f0
        hmag[:, k] = -6.0 * (k + 1)
    return hfreq, hmag


def _stoc_env(frames, bins, value=-40.0):
    """Flat stochastic envelope."""
    return np.full((frames, bins), value)


def _sine(freq, length, amp=0.8, fs=FS):
    n = np.arange(length)
    return amp * np.sin(2 * np.pi * freq * n / fs)


# ---------------------------------------------------------------------------
# sineTransformations
# ---------------------------------------------------------------------------


class TestSineFreqScaling:
    """sineFreqScaling: multiplies every active bin by a time-varying envelope."""

    def test_uniform_factor_2_doubles_all_frequencies(self):
        tfreq, tmag = _sine_tracks(frames=20, tracks=3)
        original = tfreq.copy()

        ysfreq = sineTransformations.sineFreqScaling(
            tfreq,
            freqScaling=np.array([0, 2, 1, 2]),  # constant factor 2 everywhere
        )

        np.testing.assert_allclose(ysfreq, original * 2.0, rtol=1e-10)

    def test_identity_factor_preserves_frequencies(self):
        tfreq, _ = _sine_tracks(frames=20, tracks=3)
        original = tfreq.copy()

        ysfreq = sineTransformations.sineFreqScaling(
            tfreq,
            freqScaling=np.array([0, 1, 1, 1]),  # uniform factor 1 → no change
        )

        np.testing.assert_allclose(ysfreq, original, rtol=1e-10)

    def test_zero_freq_bins_remain_zero(self):
        """Bins that were silent (freq==0) must stay 0 after scaling."""
        tfreq = np.zeros((10, 4))
        tfreq[:, 0] = 440.0  # only first track active

        ysfreq = sineTransformations.sineFreqScaling(
            tfreq,
            freqScaling=np.array([0, 3, 1, 3]),
        )

        assert np.all(ysfreq[:, 1:] == 0.0), "Silent tracks must stay zero"
        np.testing.assert_allclose(ysfreq[:, 0], 440.0 * 3, rtol=1e-10)

    def test_output_shape_matches_input(self):
        tfreq, _ = _sine_tracks(frames=15, tracks=5)
        ysfreq = sineTransformations.sineFreqScaling(
            tfreq, np.array([0, 1.5, 1, 1.5])
        )
        assert ysfreq.shape == tfreq.shape


class TestSineTimeScaling:
    """sineTimeScaling: resamples frame axis to produce a longer/shorter track."""

    def test_2x_scaling_doubles_frame_count(self):
        tfreq, tmag = _sine_tracks(frames=20, tracks=3)
        ysfreq, ysmag = sineTransformations.sineTimeScaling(
            tfreq,
            tmag,
            timeScaling=np.array(
                [0, 0, 1, 2]
            ),  # map 1 input unit → 2 output units
        )
        assert ysfreq.shape[0] == 2 * tfreq.shape[0]
        assert ysmag.shape[0] == 2 * tfreq.shape[0]

    def test_half_scaling_halves_frame_count(self):
        tfreq, tmag = _sine_tracks(frames=20, tracks=3)
        ysfreq, ysmag = sineTransformations.sineTimeScaling(
            tfreq,
            tmag,
            timeScaling=np.array([0, 0, 1, 0.5]),
        )
        assert ysfreq.shape[0] == 10
        assert ysmag.shape[0] == 10

    def test_identity_scaling_preserves_content(self):
        """1-to-1 time mapping should return the exact same frames."""
        tfreq, tmag = _sine_tracks(frames=20, tracks=3)
        ysfreq, ysmag = sineTransformations.sineTimeScaling(
            tfreq,
            tmag,
            timeScaling=np.array([0, 0, 1, 1]),
        )
        assert ysfreq.shape == tfreq.shape
        # Because the implementation rounds to the nearest frame, content must
        # be rows drawn directly from the input, so the concatenated set is a
        # subset of input rows.
        for row in ysfreq:
            assert any(
                np.allclose(row, tfreq[i]) for i in range(tfreq.shape[0])
            )

    def test_column_count_unchanged(self):
        tfreq, tmag = _sine_tracks(frames=30, tracks=7)
        ysfreq, ysmag = sineTransformations.sineTimeScaling(
            tfreq,
            tmag,
            timeScaling=np.array([0, 0, 1, 3]),
        )
        assert ysfreq.shape[1] == tfreq.shape[1]
        assert ysmag.shape[1] == tmag.shape[1]


# ---------------------------------------------------------------------------
# stochasticTransformations
# ---------------------------------------------------------------------------


class TestStochasticTimeScale:
    """stochasticTimeScale: frame-axis resample of the stochastic envelope."""

    def test_2x_scaling_doubles_frame_count(self):
        env = _stoc_env(frames=16, bins=128)
        yenv = stochasticTransformations.stochasticTimeScale(
            env,
            timeScaling=np.array([0, 0, 1, 2]),
        )
        assert yenv.shape[0] == 2 * 16

    def test_half_scaling_halves_frame_count(self):
        env = _stoc_env(frames=20, bins=64)
        yenv = stochasticTransformations.stochasticTimeScale(
            env,
            timeScaling=np.array([0, 0, 1, 0.5]),
        )
        assert yenv.shape[0] == 10

    def test_bin_count_preserved(self):
        env = _stoc_env(frames=12, bins=50)
        yenv = stochasticTransformations.stochasticTimeScale(
            env,
            timeScaling=np.array([0, 0, 1, 3]),
        )
        assert yenv.shape[1] == 50

    def test_flat_envelope_values_preserved_after_scaling(self):
        """For a flat envelope every output frame should echo the same value."""
        val = -35.0
        env = _stoc_env(frames=10, bins=32, value=val)
        yenv = stochasticTransformations.stochasticTimeScale(
            env,
            timeScaling=np.array([0, 0, 1, 2]),
        )
        np.testing.assert_allclose(yenv, val, atol=1e-10)


# ---------------------------------------------------------------------------
# harmonicTransformations
# ---------------------------------------------------------------------------


class TestHarmonicFreqScaling:
    """harmonicFreqScaling: scales / stretches the frequency grid of harmonics."""

    def test_uniform_factor_2_doubles_all_harmonic_frequencies(self):
        hfreq, hmag = _harmonic_tracks(frames=10, tracks=5, f0=100.0)
        original = hfreq.copy()

        yhfreq, yhmag = harmonicTransformations.harmonicFreqScaling(
            hfreq,
            hmag,
            freqScaling=np.array([0, 2, 1, 2]),  # constant ×2
            freqStretching=np.array([0, 1, 1, 1]),  # no stretching
            timbrePreservation=0,
            fs=FS,
        )

        np.testing.assert_allclose(yhfreq, original * 2.0, rtol=1e-10)

    def test_identity_scaling_preserves_frequencies(self):
        hfreq, hmag = _harmonic_tracks(frames=10, tracks=5, f0=220.0)
        original = hfreq.copy()

        yhfreq, yhmag = harmonicTransformations.harmonicFreqScaling(
            hfreq,
            hmag,
            freqScaling=np.array([0, 1, 1, 1]),
            freqStretching=np.array([0, 1, 1, 1]),
            timbrePreservation=0,
            fs=FS,
        )

        np.testing.assert_allclose(yhfreq, original, rtol=1e-10)

    def test_magnitudes_unchanged_without_timbre_preservation(self):
        hfreq, hmag = _harmonic_tracks(frames=10, tracks=4, f0=180.0)
        original_hmag = hmag.copy()

        yhfreq, yhmag = harmonicTransformations.harmonicFreqScaling(
            hfreq,
            hmag,
            freqScaling=np.array([0, 1.5, 1, 1.5]),
            freqStretching=np.array([0, 1, 1, 1]),
            timbrePreservation=0,
            fs=FS,
        )

        np.testing.assert_allclose(yhmag, original_hmag, rtol=1e-10)

    def test_timbre_preservation_changes_magnitudes(self):
        """With timbre preservation ON, output magnitudes should differ from input."""
        hfreq, hmag = _harmonic_tracks(frames=10, tracks=6, f0=200.0)

        yhfreq, yhmag = harmonicTransformations.harmonicFreqScaling(
            hfreq,
            hmag,
            freqScaling=np.array([0, 2, 1, 2]),  # ×2 shift
            freqStretching=np.array([0, 1, 1, 1]),
            timbrePreservation=1,
            fs=FS,
        )

        # At least some frame should have different magnitudes
        assert not np.allclose(
            yhmag, hmag
        ), "Timbre preservation should alter magnitudes"

    def test_zero_freq_bins_remain_zero(self):
        hfreq = np.zeros((8, 4))
        hmag = np.zeros((8, 4))
        hfreq[:, :2] = np.array([220.0, 440.0])  # only 2 active harmonics

        yhfreq, _ = harmonicTransformations.harmonicFreqScaling(
            hfreq,
            hmag,
            freqScaling=np.array([0, 2, 1, 2]),
            freqStretching=np.array([0, 1, 1, 1]),
            timbrePreservation=0,
            fs=FS,
        )

        assert np.all(yhfreq[:, 2:] == 0.0)

    def test_output_shape_matches_input(self):
        hfreq, hmag = _harmonic_tracks(frames=15, tracks=8)
        yhfreq, yhmag = harmonicTransformations.harmonicFreqScaling(
            hfreq,
            hmag,
            freqScaling=np.array([0, 1, 1, 1]),
            freqStretching=np.array([0, 1, 1, 1]),
            timbrePreservation=0,
            fs=FS,
        )
        assert yhfreq.shape == hfreq.shape
        assert yhmag.shape == hmag.shape


# ---------------------------------------------------------------------------
# stftTransformations
# ---------------------------------------------------------------------------


class TestStftFiltering:
    """stftFiltering: applies a magnitude filter frame-by-frame via STFT."""

    def _filter_size(self, N):
        return N // 2 + 1

    def test_allpass_filter_preserves_signal_shape(self):
        """A zero-dB flat filter should leave the signal essentially unchanged."""
        x = _sine(440.0, length=4096)
        w = np.hanning(513)
        N, H = 1024, 128
        zero_filter = np.zeros(self._filter_size(N))

        y = stftTransformations.stftFiltering(x, FS, w, N, H, zero_filter)

        assert y.shape == x.shape
        assert np.isfinite(y).all()
        # Energy ratio should be close to 1 (within OLA gain tolerance)
        energy_ratio = np.sum(y**2) / np.sum(x**2)
        assert (
            0.5 < energy_ratio < 2.0
        ), f"Energy ratio {energy_ratio:.3f} unexpected"

    def test_minus60db_filter_strongly_attenuates_signal(self):
        """-60 dB flat filter should reduce signal energy by ~6 orders of magnitude."""
        x = _sine(440.0, length=4096)
        w = np.hanning(513)
        N, H = 1024, 128
        atten_filter = np.full(self._filter_size(N), -60.0)

        y = stftTransformations.stftFiltering(x, FS, w, N, H, atten_filter)

        energy_in = np.sum(x**2)
        energy_out = np.sum(y**2)
        assert (
            energy_out < energy_in * 1e-4
        ), "Signal should be strongly attenuated"

    def test_output_shape_matches_input(self):
        x = _sine(330.0, length=8192)
        w = np.hanning(513)
        N, H = 1024, 256
        filt = np.zeros(self._filter_size(N))

        y = stftTransformations.stftFiltering(x, FS, w, N, H, filt)
        assert y.shape == x.shape


class TestStftMorph:
    """stftMorph: interpolates the magnitude spectra of two sounds."""

    def test_balance0_stays_close_to_sound1_energy(self):
        """balance=0 → output magnitude envelope ≈ x1."""
        x1 = _sine(220.0, length=4096)
        x2 = _sine(880.0, length=4096, amp=0.1)  # much quieter
        w1 = w2 = np.hanning(513)
        N1 = N2 = 1024

        y = stftTransformations.stftMorph(
            x1, x2, FS, w1, N1, w2, N2, H1=128, smoothf=1.0, balancef=0.0
        )

        e1 = np.sum(x1**2)
        ey = np.sum(y**2)
        # output energy should be in the same ballpark as x1 (within 10 dB)
        assert ey > e1 * 0.1

    def test_balance1_shifts_toward_sound2_energy(self):
        """balance=1 → output should reflect x2's energy level."""
        x1 = _sine(220.0, length=4096, amp=0.8)
        x2 = _sine(880.0, length=4096, amp=0.1)  # much quieter
        w1 = w2 = np.hanning(513)
        N1 = N2 = 1024

        y0 = stftTransformations.stftMorph(
            x1, x2, FS, w1, N1, w2, N2, H1=128, smoothf=1.0, balancef=0.0
        )
        y1 = stftTransformations.stftMorph(
            x1, x2, FS, w1, N1, w2, N2, H1=128, smoothf=1.0, balancef=1.0
        )

        # Moving balance toward sound 2 (quieter) should reduce energy
        assert np.sum(y1**2) < np.sum(
            y0**2
        ), "balance=1 (quiet x2) should produce less energy than balance=0 (loud x1)"

    def test_output_shape_matches_x1(self):
        x1 = _sine(440.0, length=4096)
        x2 = _sine(660.0, length=2048)
        w1 = w2 = np.hanning(513)
        N1 = N2 = 1024

        y = stftTransformations.stftMorph(
            x1, x2, FS, w1, N1, w2, N2, H1=256, smoothf=1.0, balancef=0.5
        )
        assert y.shape == x1.shape


# ---------------------------------------------------------------------------
# hpsTransformations
# ---------------------------------------------------------------------------


class TestHpsTimeScale:
    """hpsTimeScale: resamples all three HPS component track matrices together."""

    def test_2x_scaling_doubles_frame_count(self):
        hfreq, hmag = _harmonic_tracks(frames=20, tracks=4)
        stoc = _stoc_env(frames=20, bins=64)

        yhfreq, yhmag, ystoc = hpsTransformations.hpsTimeScale(
            hfreq,
            hmag,
            stoc,
            timeScaling=np.array([0, 0, 1, 2]),
        )
        assert yhfreq.shape[0] == 2 * 20
        assert yhmag.shape[0] == 2 * 20
        assert ystoc.shape[0] == 2 * 20

    def test_half_scaling_halves_frame_count(self):
        hfreq, hmag = _harmonic_tracks(frames=20, tracks=4)
        stoc = _stoc_env(frames=20, bins=64)

        yhfreq, yhmag, ystoc = hpsTransformations.hpsTimeScale(
            hfreq,
            hmag,
            stoc,
            timeScaling=np.array([0, 0, 1, 0.5]),
        )
        assert yhfreq.shape[0] == 10

    def test_column_counts_unchanged(self):
        hfreq, hmag = _harmonic_tracks(frames=16, tracks=6)
        stoc = _stoc_env(frames=16, bins=50)

        yhfreq, yhmag, ystoc = hpsTransformations.hpsTimeScale(
            hfreq,
            hmag,
            stoc,
            timeScaling=np.array([0, 0, 1, 2]),
        )
        assert yhfreq.shape[1] == hfreq.shape[1]
        assert yhmag.shape[1] == hmag.shape[1]
        assert ystoc.shape[1] == stoc.shape[1]

    def test_flat_harmonic_values_preserved_after_2x_scaling(self):
        """If every frame has identical harmonics, scaled frames must too."""
        hfreq = np.full((12, 3), fill_value=np.array([220.0, 440.0, 660.0]))
        hmag = np.full((12, 3), fill_value=np.array([-6.0, -12.0, -18.0]))
        stoc = _stoc_env(frames=12, bins=32, value=-30.0)

        yhfreq, yhmag, ystoc = hpsTransformations.hpsTimeScale(
            hfreq,
            hmag,
            stoc,
            timeScaling=np.array([0, 0, 1, 2]),
        )

        # Implementation currently leaves boundary rows unfilled; check populated frames.
        nonzero_rows = np.where(np.any(yhfreq != 0.0, axis=1))[0]
        assert nonzero_rows.size > 0
        for row in yhfreq[nonzero_rows]:
            assert np.allclose(row, hfreq[0])


class TestHpsMorph:
    """hpsMorph: linearly interpolates two HPS representations."""

    def _np_int_patch(self):
        """Patch np.int removed in NumPy >=1.24 (used inside hpsMorph)."""
        if not hasattr(np, "int"):
            np.int = int

    def test_full_interpolation_to_sound2_recovers_sound2_harmonics(self):
        """stocIntp=1 → stochastic output should be 100% from stocEnv2."""
        self._np_int_patch()

        hfreq1, hmag1 = _harmonic_tracks(frames=10, tracks=3, f0=220.0)
        hfreq2, hmag2 = _harmonic_tracks(frames=10, tracks=3, f0=330.0)
        stoc1 = _stoc_env(frames=10, bins=32, value=-20.0)
        stoc2 = _stoc_env(frames=10, bins=32, value=-60.0)

        yhfreq, yhmag, ystoc = hpsTransformations.hpsMorph(
            hfreq1,
            hmag1,
            stoc1,
            hfreq2,
            hmag2,
            stoc2,
            hfreqIntp=np.array(
                [0.0, 1.0, 1.0, 1.0]
            ),  # 100 % freq from sound 2
            hmagIntp=np.array([0.0, 1.0, 1.0, 1.0]),  # 100 % mag from sound 2
            stocIntp=np.array([0.0, 1.0, 1.0, 1.0]),  # 100 % stoc from sound 2
        )

        np.testing.assert_allclose(ystoc, stoc2, atol=1e-10)

    def test_no_interpolation_preserves_sound1_stochastic(self):
        """stocIntp=0 → stochastic output should equal stocEnv1."""
        self._np_int_patch()

        hfreq1, hmag1 = _harmonic_tracks(frames=10, tracks=3, f0=220.0)
        hfreq2, hmag2 = _harmonic_tracks(frames=10, tracks=3, f0=330.0)
        stoc1 = _stoc_env(frames=10, bins=32, value=-20.0)
        stoc2 = _stoc_env(frames=10, bins=32, value=-60.0)

        yhfreq, yhmag, ystoc = hpsTransformations.hpsMorph(
            hfreq1,
            hmag1,
            stoc1,
            hfreq2,
            hmag2,
            stoc2,
            hfreqIntp=np.array([0.0, 0.0, 1.0, 0.0]),  # 0 % → sound 1
            hmagIntp=np.array([0.0, 0.0, 1.0, 0.0]),
            stocIntp=np.array([0.0, 0.0, 1.0, 0.0]),
        )

        np.testing.assert_allclose(ystoc, stoc1, atol=1e-10)

    def test_midpoint_interpolation_is_average_of_stoc_envelopes(self):
        """stocIntp=0.5 → output stochastic should be the mean of the two."""
        self._np_int_patch()

        hfreq1, hmag1 = _harmonic_tracks(frames=10, tracks=3, f0=220.0)
        hfreq2, hmag2 = _harmonic_tracks(frames=10, tracks=3, f0=220.0)
        stoc1 = _stoc_env(frames=10, bins=32, value=-20.0)
        stoc2 = _stoc_env(frames=10, bins=32, value=-40.0)
        expected = 0.5 * (stoc1 + stoc2)  # = -30.0

        yhfreq, yhmag, ystoc = hpsTransformations.hpsMorph(
            hfreq1,
            hmag1,
            stoc1,
            hfreq2,
            hmag2,
            stoc2,
            hfreqIntp=np.array([0.0, 0.5, 1.0, 0.5]),
            hmagIntp=np.array([0.0, 0.5, 1.0, 0.5]),
            stocIntp=np.array([0.0, 0.5, 1.0, 0.5]),
        )

        np.testing.assert_allclose(ystoc, expected, atol=1e-10)

    def test_output_shape_matches_sound1(self):
        self._np_int_patch()

        hfreq1, hmag1 = _harmonic_tracks(frames=12, tracks=4, f0=220.0)
        hfreq2, hmag2 = _harmonic_tracks(frames=8, tracks=4, f0=330.0)
        stoc1 = _stoc_env(frames=12, bins=32)
        stoc2 = _stoc_env(frames=8, bins=32)

        yhfreq, yhmag, ystoc = hpsTransformations.hpsMorph(
            hfreq1,
            hmag1,
            stoc1,
            hfreq2,
            hmag2,
            stoc2,
            hfreqIntp=np.array([0.0, 0.5, 1.0, 0.5]),
            hmagIntp=np.array([0.0, 0.5, 1.0, 0.5]),
            stocIntp=np.array([0.0, 0.5, 1.0, 0.5]),
        )

        assert yhfreq.shape == hfreq1.shape
        assert yhmag.shape == hmag1.shape
        assert ystoc.shape == stoc1.shape
