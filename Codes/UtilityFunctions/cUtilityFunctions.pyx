#this is a cython wrapper on C functions to call them in python

import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from cUtilityFunctions cimport *

np.import_array()

def genbh92lobe(x):
	"comments"
	
	cdef np.ndarray[np.float_t, ndim=1] x_arr
	cdef np.ndarray[np.float_t, ndim=1] y_arr
	
	
	x_arr = np.ascontiguousarray(x, dtype=np.float)
	y_arr = np.empty((x_arr.shape[0],), dtype=np.float)
	
	genbh92lobe_C(<double *>x_arr.data,<double *>y_arr.data, x_arr.shape[0])

	
	return y_arr
	
	

	
def genspecsines(iploc, ipmag, ipphase,N):
	"comments"
	
	cdef np.ndarray[np.float_t, ndim=1] iploc_arr
	cdef np.ndarray[np.float_t, ndim=1] ipmag_arr
	cdef np.ndarray[np.float_t, ndim=1] ipphase_arr
	cdef np.ndarray[np.float_t, ndim=1] real_arr
	cdef np.ndarray[np.float_t, ndim=1] imag_arr
		
	iploc_arr = np.ascontiguousarray(iploc, dtype=np.float)
	ipmag_arr = np.ascontiguousarray(ipmag, dtype=np.float)
	ipphase_arr = np.ascontiguousarray(ipphase, dtype=np.float)
	
	real_arr = np.zeros((N,), dtype=np.float)
	imag_arr = np.zeros((N,), dtype=np.float)
		
	genspecsines_C(<double *>iploc_arr.data, <double *>ipmag_arr.data, <double *>ipphase_arr.data, iploc_arr.shape[0],  <double *>real_arr.data,  <double *>imag_arr.data, N)
	
	out = real_arr.astype(complex)
	out.imag = imag_arr
	return out
