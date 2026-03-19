from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


def hpsTimeScale(
    hfreq: np.ndarray,
    hmag: np.ndarray,
    stocEnv: np.ndarray,
    timeScaling: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time scaling of the harmonic plus stochastic representation.

    Args:
        hfreq: Harmonic frequencies (frames x harmonics)
        hmag: Harmonic magnitudes (frames x harmonics)
        stocEnv: Stochastic envelope (frames x bins)
        timeScaling: Scaling factors (time-value pairs)

    Returns:
        yhfreq: Output harmonic frequencies
        yhmag: Output harmonic magnitudes
        ystocEnv: Output stochastic envelope
    """
    if timeScaling.size % 2 != 0:
        raise ValueError("Time scaling array does not have an even size")
    L = hfreq[:, 0].size
    maxInTime = max(timeScaling[::2])
    maxOutTime = max(timeScaling[1::2])
    outL = int(L * maxOutTime / maxInTime)
    inFrames = (L - 1) * timeScaling[::2] / maxInTime
    outFrames = outL * timeScaling[1::2] / maxOutTime
    timeScalingEnv = interp1d(outFrames, inFrames, fill_value=0)
    indexes = timeScalingEnv(np.arange(outL))
    yhfreq = np.zeros((indexes.shape[0], hfreq.shape[1]))
    yhmag = np.zeros((indexes.shape[0], hmag.shape[1]))
    ystocEnv = np.zeros((indexes.shape[0], stocEnv.shape[1]))
    for frameIdx, idx_val in enumerate(indexes[1:]):
        idx = int(round(idx_val))
        yhfreq[frameIdx, :] = hfreq[idx, :]
        yhmag[frameIdx, :] = hmag[idx, :]
        ystocEnv[frameIdx, :] = stocEnv[idx, :]
    return yhfreq, yhmag, ystocEnv


def hpsMorph(
    hfreq1: np.ndarray,
    hmag1: np.ndarray,
    stocEnv1: np.ndarray,
    hfreq2: np.ndarray,
    hmag2: np.ndarray,
    stocEnv2: np.ndarray,
    hfreqIntp: np.ndarray,
    hmagIntp: np.ndarray,
    stocIntp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Morph between two sounds using the harmonic plus stochastic model.

    Args:
        hfreq1, hmag1, stocEnv1: HPS representation of sound 1
        hfreq2, hmag2, stocEnv2: HPS representation of sound 2
        hfreqIntp: Interpolation factor for harmonic frequencies (time-value pairs)
        hmagIntp: Interpolation factor for harmonic magnitudes (time-value pairs)
        stocIntp: Interpolation factor for stochastic component (time-value pairs)

    Returns:
        yhfreq: Output harmonic frequencies
        yhmag: Output harmonic magnitudes
        ystocEnv: Output stochastic envelope
    """
    if hfreqIntp.size % 2 != 0:
        raise ValueError(
            "Harmonic frequencies interpolation array does not have an even size"
        )
    if hmagIntp.size % 2 != 0:
        raise ValueError(
            "Harmonic magnitudes interpolation does not have an even size"
        )
    if stocIntp.size % 2 != 0:
        raise ValueError(
            "Stochastic component array does not have an even size"
        )
    L1 = hfreq1[:, 0].size
    L2 = hfreq2[:, 0].size
    hfreqIntp[::2] = (L1 - 1) * hfreqIntp[::2] / hfreqIntp[-2]
    hmagIntp[::2] = (L1 - 1) * hmagIntp[::2] / hmagIntp[-2]
    stocIntp[::2] = (L1 - 1) * stocIntp[::2] / stocIntp[-2]
    hfreqIntpEnv = interp1d(hfreqIntp[::2], hfreqIntp[1::2], fill_value=0)
    hfreqIndexes = hfreqIntpEnv(np.arange(L1))
    hmagIntpEnv = interp1d(hmagIntp[::2], hmagIntp[1::2], fill_value=0)
    hmagIndexes = hmagIntpEnv(np.arange(L1))
    stocIntpEnv = interp1d(stocIntp[::2], stocIntp[1::2], fill_value=0)
    stocIndexes = stocIntpEnv(np.arange(L1))
    yhfreq = np.zeros_like(hfreq1)
    yhmag = np.zeros_like(hmag1)
    ystocEnv = np.zeros_like(stocEnv1)
    for idx in range(L1):
        dataIndex = int(round(((L2 - 1) * idx) / float(L1 - 1)))
        harmonics = np.intersect1d(
            np.nonzero(hfreq1[idx, :])[0],
            np.nonzero(hfreq2[dataIndex, :])[0],
        )
        yhfreq[idx, harmonics] = (1 - hfreqIndexes[idx]) * hfreq1[
            idx, harmonics
        ] + hfreqIndexes[idx] * hfreq2[dataIndex, harmonics]
        yhmag[idx, harmonics] = (1 - hmagIndexes[idx]) * hmag1[
            idx, harmonics
        ] + hmagIndexes[idx] * hmag2[dataIndex, harmonics]
        ystocEnv[idx, :] = (1 - stocIndexes[idx]) * stocEnv1[
            idx, :
        ] + stocIndexes[idx] * stocEnv2[dataIndex, :]
    return yhfreq, yhmag, ystocEnv
