from setuptools import setup, find_packages
from distutils.core import Extension
from distutils.sysconfig import *
from distutils.util import *
from Cython.Distutils import build_ext
import numpy
import os
import os.path

py_inc = [get_python_inc()]

np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]
ext_inc = os


sourcefiles = ["sms_tools/models/utilFunctions_C/utilFunctions.c",
               "sms_tools/models/utilFunctions_C/cutilFunctions.pyx"]

setup(
    name='sms_tools',
    version="0.1",
    packages=find_packages(),
    scripts=['bin/models_GUI', 'bin/transformations_GUI', 'bin/startAssignment'],
    install_requires=['ipython', 'numpy', 'scipy', 'matplotlib', 'cython', 'requests'],
    cmdclass={'build_ext': build_ext},
    ext_modules=[Extension("utilFunctions_C",
                 sourcefiles, libraries=['m'], include_dirs=py_inc + np_inc)],
    author="Xavier Serra",
    license="Affero GPL"
)
