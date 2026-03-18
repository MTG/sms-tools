sms-tools
========= 

Sound analysis/synthesis tools for music applications written in python.

The package includes the following sound analysis/synthesis models:

* dftModel.py: models based on the Discrete Fourier Transform
* stft.py: models based on the Short-Time Fourier Transform
* sineModel.py: models based on a Sinusoidal Model
* harmonicModel.py: models based on a Harmonic Model
* stochasticModel.py: models based on a Stochastic Model
* sprModel.py: models based on a Sinusoidal plus Residual Model
* spsModel.py: models based on a Sinusoidal plus Stochastic Model
* hprModel.py: models based on a Harmonic plus Residual Model
* hpsModel.py: models based on a Harmonic plus Stochastic Model


Installation
------------

Install using pip:

    pip install sms-tools


When installing via pip, the Cython extension is built automatically if a compatible compiler and Python environment are available. This provides significant speedups for core routines.

    pip install sms-tools

If you are developing locally or want to ensure the Cython extension is built, you can run:

    pip install sms-tools
    python setup.py build_ext --inplace

If you encounter issues with the Cython extension, ensure you have Cython, setuptools, and a C compiler installed. The extension is optional; sms-tools will fall back to pure-Python routines if unavailable.

You can verify which backend is active at runtime:

    python - <<'PY'
    from smstools.models import utilFunctions as UF
    print("Using Cython backend:", UF.UF_C is not None)
    print("Backend module:", getattr(UF.UF_C, "__file__", None))
    PY

Binary packages are available for Linux, macOS (Intel & Apple Silicon) and Windows (64 bit) on all recent python versions.

For details about automatic Cython acceleration and fallback behavior, see the
"Cython backend" section below.

To build and install the package locally you can use the python packaging tools:

    pip install build
    python -m build


Cython backend
--------------

sms-tools includes a compiled Cython extension for selected core routines.
When available, it is used automatically for better performance.
If it cannot be imported, sms-tools falls back to pure-Python implementations
with the same public behavior (typically slower).

You can verify which backend is active at runtime:

    python - <<'PY'
    from smstools.models import utilFunctions as UF
    print("Using Cython backend:", UF.UF_C is not None)
    print("Backend module:", getattr(UF.UF_C, "__file__", None))
    PY

Testing
-------

To install test dependencies and run the test suite, use:

    pip install .[test]
    pytest

You can run a specific test file with:

    pytest -v tests/test_cython_vs_python.py

This will execute all tests and print detailed output.

sms-tools-materials repository
-------

There is a separate repository containing teaching materials, example notebooks, and exercises used in courses that use the sms-tools package. It includes Jupyter notebooks, audio examples, and practical guides to help students and researchers learn about sound analysis, synthesis, and transformation with sms-tools.

For more information and resources, see: https://github.com/MTG/sms-tools-materials

License
-------

sms-tools is made available under the terms of the Affero GPL license (http://www.gnu.org/licenses/agpl-3.0.en.html). 
