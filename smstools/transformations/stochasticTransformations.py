
"""
Transformations using the stochasticModel (stochastic spectral model).
"""

from typing import Tuple

import numpy as np
from scipy.interpolate import interp1d


def stochasticTimeScale(
    stocEnv: np.ndarray,
    timeScaling: np.ndarray,
) -> np.ndarray:
    """
    Time scaling of the stochastic representation of a sound.

    Args:
        stocEnv: Stochastic envelope (frames x bins)
        timeScaling: Scaling factors (time-value pairs)

    Returns:
        ystocEnv: Output stochastic envelope (frames x bins)
    """
    if timeScaling.size % 2 != 0:
        raise ValueError("Time scaling array does not have an even size")
    L = stocEnv.shape[0]
    outL = int(L * timeScaling[-1] / timeScaling[-2])
    timeScalingEnv = interp1d(
        timeScaling[::2] / timeScaling[-2], timeScaling[1::2] / timeScaling[-1]
    )
    indexes = (L - 1) * timeScalingEnv(np.arange(outL) / float(outL))
    ystocEnv = stocEnv[0, :]
    for l in indexes[1:]:
        ystocEnv = np.vstack((ystocEnv, stocEnv[int(round(l)), :]))
    return ystocEnv
