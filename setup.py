from setuptools import Extension, setup
from Cython.Build import cythonize
import sys
import numpy

sourcefiles = [
    "smstools/models/utilFunctions_C/utilFunctions.c",
    "smstools/models/utilFunctions_C/cutilFunctions.pyx"
]
if sys.platform == 'win32':
    library = 'msvcrt'
else:
    library = 'm'

extensions = [
    Extension(
        "utilFunctions_C",
        sourcefiles,
        include_dirs=[numpy.get_include()],
        libraries=[library]
    ),
]
setup(
    name="SMS Tools",
    ext_modules=cythonize(extensions),
)