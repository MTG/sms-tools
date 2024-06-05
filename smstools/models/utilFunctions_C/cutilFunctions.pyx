#this is a cython wrapper on C functions to call them in python

import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from cutilFunctions cimport *

np.import_array()


def twm(pfreq, pmag, f0c):
    """This is a cython wrapper for a C function which is bit exact with the python version of this function
       For information about the input arguments please refere to the original python function
    """
    
    
    cdef np.ndarray[np.float_t, ndim=1] f0_arr
    cdef np.ndarray[np.float_t, ndim=1] f0Error_arr
    cdef np.ndarray[np.float_t, ndim=1] pfreq_arr
    cdef np.ndarray[np.float_t, ndim=1] pmag_arr
    cdef np.ndarray[np.float_t, ndim=1] f0c_arr
    
    f0_arr = np.ascontiguousarray(np.array([-1]), dtype=float)
    f0Error_arr = np.ascontiguousarray(np.array([-1]), dtype=float)

    pfreq_arr = np.ascontiguousarray(pfreq, dtype=float)
    pmag_arr = np.ascontiguousarray(pmag, dtype=float)
    f0c_arr = np.ascontiguousarray(f0c, dtype=float)

    TWM_C(<double*>pfreq_arr.data, <double *>pmag_arr.data, pfreq_arr.shape[0], <double *>f0c_arr.data, f0c_arr.shape[0], <double*>f0_arr.data, <double*>f0Error_arr.data)

    return f0_arr[0], f0Error_arr[0]


def genbh92lobe(x):
    "comments"
    
    cdef np.ndarray[np.float_t, ndim=1] x_arr
    cdef np.ndarray[np.float_t, ndim=1] y_arr
    
    
    x_arr = np.ascontiguousarray(x, dtype=float)
    y_arr = np.empty((x_arr.shape[0],), dtype=float)
    
    genbh92lobe_C(<double *>x_arr.data,<double *>y_arr.data, x_arr.shape[0])

    
    return y_arr
    
    
    
def genSpecSines(iploc, ipmag, ipphase,N):
    "comments"
    
    cdef np.ndarray[np.float_t, ndim=1] iploc_arr
    cdef np.ndarray[np.float_t, ndim=1] ipmag_arr
    cdef np.ndarray[np.float_t, ndim=1] ipphase_arr
    cdef np.ndarray[np.float_t, ndim=1] real_arr
    cdef np.ndarray[np.float_t, ndim=1] imag_arr
        
    iploc_arr = np.ascontiguousarray(iploc, dtype=float)
    ipmag_arr = np.ascontiguousarray(ipmag, dtype=float)
    ipphase_arr = np.ascontiguousarray(ipphase, dtype=float)
    
    real_arr = np.zeros((N,), dtype=float)
    imag_arr = np.zeros((N,), dtype=float)
        
    genspecsines_C(<double *>iploc_arr.data, <double *>ipmag_arr.data, <double *>ipphase_arr.data, iploc_arr.shape[0],  <double *>real_arr.data,  <double *>imag_arr.data, N)
    
    out = real_arr.astype(complex)
    out.imag = imag_arr
    return out
