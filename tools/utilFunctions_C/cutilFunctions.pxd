#this is a cython wrapper on C functions to call them in python


cdef extern from "utilFunctions.h":
	
	int TWM_C(double *pfreq, double *pmag, int nPeaks, double *f0c, int nf0c, double *f0, double *f0error)
	void genbh92lobe_C(double *x, double *y, int N)
	void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec)