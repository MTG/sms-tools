# Sound Transformations Module

This directory contains all the software for transforming sounds using the various analysis/synthesis models implemented in the SMS Tools package.

## Overview
The transformation modules operate on the outputs of models such as sineModel, harmonicModel, stochasticModel, hpsModel, and others. They provide algorithms for modifying, morphing, and resynthesizing audio in the time-frequency domain.

## Main Modules
- `harmonicTransformations.py`: Transformations for harmonic model outputs (e.g., pitch shifting, time scaling, timbre modification).
- `hpsTransformations.py`: Transformations for harmonic plus stochastic model outputs.
- `sineTransformations.py`: Transformations for sine model outputs (e.g., frequency scaling, time stretching).
- `stftTransformations.py`: Transformations using the short-time Fourier transform (STFT).
- `stochasticTransformations.py`: Transformations for stochastic model outputs.

## Usage
Each module provides functions that take model analysis outputs (e.g., frequency, magnitude, phase tracks) and apply transformations such as:
- Time scaling (stretching/compressing)
- Frequency scaling (pitch shifting)
- Morphing between two sounds
- Timbre modification

See the docstrings in each module for function-level documentation and usage examples.

## Requirements
- Numpy, Scipy
- Use in conjunction with the analysis/synthesis models in `../models/`

## References
- Serra, X. (1997). "Musical Sound Modeling with Sinusoids plus Noise". In Musical Signal Processing, Swets & Zeitlinger.
- SMS Tools documentation: https://www.upf.edu/web/mtg/sms-tools