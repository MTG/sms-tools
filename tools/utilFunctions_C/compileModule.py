from distutils.core import setup, Extension
from distutils.sysconfig import *
from distutils.util import *
from Cython.Distutils import build_ext
import numpy
import os
import os.path

try:
   from distutils.command.build_py import build_py_2to3 \
       as build_py
except ImportError:
   from distutils.command.build_py import build_py
   
try:
   from Cython.Distutils import build_ext
except ImportError:
   use_cython = False
else:
   use_cython = True
   

py_inc = [get_python_inc()]

np_lib = os.path.dirname(numpy.__file__)
np_inc = [os.path.join(np_lib, 'core/include')]
ext_inc = os

sourcefiles = ["utilFunctions.c", "cutilFunctions.pyx"]

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("utilFunctions_C",sourcefiles, libraries=['m'], include_dirs=py_inc + np_inc)]
  )
