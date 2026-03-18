from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


def sineTimeScaling(
    sfreq: np.ndarray,
    smag: np.ndarray,
    timeScaling: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Time scaling of sinusoidal tracks.

    Args:
        sfreq: Frequencies of input sinusoidal tracks (frames x sines)
        smag: Magnitudes of input sinusoidal tracks (frames x sines)
        timeScaling: Scaling factors (time-value pairs)

    Returns:
        ysfreq: Output frequencies (frames x sines)
        ysmag: Output magnitudes (frames x sines)
    """
    if timeScaling.size % 2 != 0:
        raise ValueError("Time scaling array does not have an even size")
    L = sfreq.shape[0]
    maxInTime = max(timeScaling[::2])
    maxOutTime = max(timeScaling[1::2])
    outL = int(L * maxOutTime / maxInTime)
    inFrames = (L - 1) * timeScaling[::2] / maxInTime
    outFrames = outL * timeScaling[1::2] / maxOutTime
    timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)
    indexes = timeScalingEnv(np.arange(outL))
    ysfreq = sfreq[int(round(indexes[0])), :]
    ysmag = smag[int(round(indexes[0])), :]
    for l in indexes[1:]:
        ysfreq = np.vstack((ysfreq, sfreq[int(round(l)), :]))
        ysmag = np.vstack((ysmag, smag[int(round(l)), :]))
    return ysfreq, ysmag



def sineFreqScaling(
    sfreq: np.ndarray,
    freqScaling: np.ndarray,
) -> np.ndarray:
    """
    Frequency scaling of sinusoidal tracks.

    Args:
        sfreq: Frequencies of input sinusoidal tracks (frames x sines)
        freqScaling: Scaling factors (time-value pairs, value of 1 is no scaling)

    Returns:
        ysfreq: Output frequencies (frames x sines)
    """
    if freqScaling.size % 2 != 0:
        raise ValueError("Frequency scaling array does not have an even size")
    L = sfreq.shape[0]
    freqScalingEnv = np.interp(
        np.arange(L), L * freqScaling[::2] / freqScaling[-2], freqScaling[1::2]
    )
    ysfreq = np.zeros_like(sfreq)
    for l in range(L):
        ind_valid = np.where(sfreq[l, :] != 0)[0]
        if ind_valid.size == 0:
            continue
        ysfreq[l, ind_valid] = sfreq[l, ind_valid] * freqScalingEnv[l]
    return ysfreq
