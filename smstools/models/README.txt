# Spectral Models Module

This directory contains the implementation of all spectral analysis/synthesis models used in the SMS Tools package.

## Overview
The models in this directory decompose, analyze, and resynthesize audio signals using various spectral techniques. Each model is implemented as a Python module with analysis, synthesis, and full model functions.

## Main Models
- `dftModel.py`: Discrete Fourier Transform (DFT) analysis and synthesis functions.
- `sineModel.py`: Sinusoidal model for tracking and synthesizing sinusoidal components.
- `harmonicModel.py`: Harmonic model for extracting and resynthesizing harmonic partials.
- `stochasticModel.py`: Stochastic model for modeling the noise-like residual.
- `sprModel.py`: Sinusoidal plus residual (SPR) model, combining sinusoidal and noise components.
- `spsModel.py`: Sinusoidal plus stochastic (SPS) model.
- `hprModel.py`: Harmonic plus residual (HPR) model.
- `hpsModel.py`: Harmonic plus stochastic (HPS) model.
- `stft.py`: Short-Time Fourier Transform utilities.
- `utilFunctions.py`: Utility functions for windowing, synthesis, and other signal processing tasks.

## Usage
- Each model provides `*Anal`, `*Synth`, and full model functions (e.g., `sineModelAnal`, `sineModelSynth`, `sineModel`).
- Use these modules as building blocks for sound analysis, transformation, and synthesis.
- See the docstrings in each file for detailed API documentation and usage examples.

## Requirements
- Numpy, Scipy

## References
- Serra, X. (1997). "Musical Sound Modeling with Sinusoids plus Noise". In Musical Signal Processing, Swets & Zeitlinger.
- SMS Tools documentation: https://www.upf.edu/web/mtg/sms-tools