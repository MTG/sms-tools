#this is a cython wrapper on C functions to call them in python


cdef extern from "twm.h":
	
	double f0DetectionTwm_C(double *ploc, double *pmag, int nPeaks, int N, int fs, double ef0max, double minf0, double maxf0, int maxnpeaks)