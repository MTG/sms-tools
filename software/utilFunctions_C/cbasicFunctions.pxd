#this is a cython wrapper on C functions to call them in python


cdef extern from "basicFunctions.h":
	
	void genbh92lobe_C(double *x, double *y, int N)
	void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec)