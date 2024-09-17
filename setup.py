import sys

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup

sourcefiles = [
    "smstools/models/utilFunctions_C/utilFunctions.c",
    "smstools/models/utilFunctions_C/cutilFunctions.pyx",
]
if sys.platform == "win32":
    library = "msvcrt"
else:
    library = "m"

extensions = [
    Extension(
        "smstools.models.utilFunctions_C.utilFunctions_C",
        sourcefiles,
        include_dirs=[numpy.get_include()],
        libraries=[library],
    ),
]
setup(
    name="SMS Tools",
    ext_modules=cythonize(extensions),
)
