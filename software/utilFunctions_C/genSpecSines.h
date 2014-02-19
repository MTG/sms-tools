
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <math.h>
//#include "tables.h"


#ifndef GENSPECSINES_H
#define GENSPECSINES_H

void genbh92lobe_C(double *x, double *y, int N);
void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec);
#endif