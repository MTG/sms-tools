#this is a cython wrapper on C functions to call them in python

import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from ctwm cimport *

np.import_array()


def f0DetectionTwm(ploc, pmag, N, fs, ef0max, minf0, maxf0, maxnpeaks=10):
    """This is a cython wrapper for a C function which is bit exact with the python version of this function
       For information about the input arguments please refere to the original python function
    """
    
    
    cdef np.ndarray[np.float_t, ndim=1] ploc_arr
    cdef np.ndarray[np.float_t, ndim=1] pmag_arr
    
    ploc_arr = np.ascontiguousarray(ploc, dtype=np.float)
    pmag_arr = np.ascontiguousarray(pmag, dtype=np.float)

    f0 = f0DetectionTwm_C(<double*>ploc_arr.data, <double *>pmag_arr.data, ploc_arr.shape[0], N, fs, ef0max, minf0, maxf0, maxnpeaks)

    return f0
    