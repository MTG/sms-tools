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

Binary packages are available for Linux, macOS (Intel & Apple Silicon) and Windows (64 bit) on all recent python versions.

To build and install the package locally you can use the python packaging tools:

    pip install build
    python -m build


Running tests
-------------

To run the unit test suite locally:

    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install -e ".[test]"
    python -m pytest

To run a single test file:

    python -m pytest tests/test_errors.py

To run a single test function by name:

    python -m pytest tests/test_errors.py -k wavread

Main smoke test files:

    tests/test_models_smoke.py
    tests/test_transformations_smoke.py

Run only smoke tests:

    python -m pytest -k smoke


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

Test case summary:

* `tests/test_api_contracts.py`: API/signature and output-shape contract checks for core model entry points.
* `tests/test_errors.py`: error-handling contracts (invalid parameters and invalid I/O paths).
* `tests/test_models_smoke.py`: fast smoke coverage for all analysis/synthesis model modules.
* `tests/test_transformations_smoke.py`: fast smoke coverage for all transformation modules.
* `tests/test_models_ground_truth.py`: algorithmic/ground-truth model tests on synthetic signals (frequency accuracy, additivity, and quality invariants).
* `tests/test_transformations_ground_truth.py`: algorithmic/ground-truth transformation tests (scaling/morphing identity behavior and expected interpolation/attenuation trends).


Jupyter Notebooks
-------
We provide a separate repository of examples and teaching materials in the form of Jupyter notebooks.
You can find them at https://github.com/MTG/sms-tools-materials

License
-------

sms-tools is made available under the terms of the Affero GPL license (http://www.gnu.org/licenses/agpl-3.0.en.html). 
