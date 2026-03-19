from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


def harmonicFreqScaling(
    hfreq: np.ndarray,
    hmag: np.ndarray,
    freqScaling: np.ndarray,
    freqStretching: np.ndarray,
    timbrePreservation: int,
    fs: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Frequency scaling and stretching of the harmonics of a sound.

    Args:
        hfreq: Harmonic frequencies (frames x harmonics)
        hmag: Harmonic magnitudes (frames x harmonics)
        freqScaling: Scaling factors (time-value pairs)
        freqStretching: Stretching factors (time-value pairs)
        timbrePreservation: 0 (no preservation), 1 (preserve timbre)
        fs: Sampling rate

    Returns:
        yhfreq: Output harmonic frequencies
        yhmag: Output harmonic magnitudes
    """
    if freqScaling.size % 2 != 0:
        raise ValueError("Frequency scaling array does not have an even size")
    if freqStretching.size % 2 != 0:
        raise ValueError(
            "Frequency stretching array does not have an even size"
        )
    L = hfreq.shape[0]
    freqScalingEnv = np.interp(
        np.arange(L), L * freqScaling[::2] / freqScaling[-2], freqScaling[1::2]
    )
    freqStretchingEnv = np.interp(
        np.arange(L),
        L * freqStretching[::2] / freqStretching[-2],
        freqStretching[1::2],
    )
    yhfreq = np.zeros_like(hfreq)
    yhmag = np.zeros_like(hmag)
    for idx in range(L):
        ind_valid = np.where(hfreq[idx, :] != 0)[0]
        if ind_valid.size == 0:
            continue
        if (timbrePreservation == 1) and (ind_valid.size > 1):
            x_vals = np.append(np.append(0, hfreq[idx, ind_valid]), fs / 2)
            y_vals = np.append(
                np.append(hmag[idx, 0], hmag[idx, ind_valid]), hmag[idx, -1]
            )
            specEnvelope = interp1d(
                x_vals,
                y_vals,
                kind="linear",
                bounds_error=False,
                fill_value=-100,
            )
        yhfreq[idx, ind_valid] = hfreq[idx, ind_valid] * freqScalingEnv[idx]
        yhfreq[idx, ind_valid] = yhfreq[idx, ind_valid] * (
            freqStretchingEnv[idx] ** ind_valid
        )
        if (timbrePreservation == 1) and (ind_valid.size > 1):
            yhmag[idx, ind_valid] = specEnvelope(yhfreq[idx, ind_valid])
        else:
            yhmag[idx, ind_valid] = hmag[idx, ind_valid]
    return yhfreq, yhmag
