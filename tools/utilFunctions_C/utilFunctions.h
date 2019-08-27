
#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
#include <math.h>

#ifndef UTILFUNCTIONS_H 
    
#define UTILFUNCTIONS_H

#define BH_SIZE 1001
#define BH_SIZE_BY2 501
#define MFACTOR 100


#ifndef max
    #define max( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef min
    #define min( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define oneTOtwo(i,j, nCol) ((i)*nCol + (j))

#define MAGSILENCE -120
#define F0CANDPERPEAK 6
#define F0NUMPEAK 3                //number of top peaks which should be considerd for generating f0 candidates
#define TWM_p 0.5                  //weighting by frequency value
#define TWM_q 1.4                  //weighting related to magnitude of peaks
#define TWM_r 0.5                  //scaling related to magnitude of peaks
#define TWM_rho 0.33               //weighting of MP error
#define MAXNPEAKS 10               // maximum number of peaks used for TWM 

void genbh92lobe_C(double *x, double *y, int N);
void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec);
void maxValArg(double *data, int dLen, double *max_val, int*max_ind);
void minValArg(double *data, int dLen, double *min_val, int*min_ind);
void computeTWMError(double **peakMTX1, int nCols1, double **peakMTX2, int nCols2, int maxnpeaks, double * pmag, double *f0Error, int nF0Cands, int PMorMP);
int nearestElement(double val, double *data, int len, double *min_val);
int TWM_C(double *pfreq, double *pmag, int nPeaks, double *f0c, int nf0c, double *f0, double *f0error);

#endif  //UTILFUNCTIONS_H