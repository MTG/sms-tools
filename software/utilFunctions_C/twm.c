

#include "twm.h"


void maxValArg(double *data, int dLen, double *max_val, int*max_ind)
{
    int ii;
    *max_val = -1*FLT_MAX;
    for(ii=0; ii<dLen;ii++)
    {
        if(*max_val < data[ii])
        {
            *max_val = data[ii];
            *max_ind = ii;
        }
    }
}

void minValArg(double *data, int dLen, double *min_val, int*min_ind)
{
    int ii;
    *min_val = FLT_MAX;
    for(ii=0; ii<dLen;ii++)
    {
        if(*min_val > data[ii])
        {
            *min_val = data[ii];
            *min_ind = ii;
        }
    }
}



double f0DetectionTwm_C(double *pfreq1, double *pmag, int nPeaks, int N, int fs, double ef0max, double minf0, double maxf0, int maxnpeaks)
{
    double f0=0, f0error;
    int ii,jj, min_ind, max_ind, f0CandCnt=0;
    double *pfreq, *pmag_temp;
    double min_val, max_val;
    double *f0Cands, f0down;

    pfreq = (double*)malloc(sizeof(double)*nPeaks);
	pmag_temp = (double*)malloc(sizeof(double)*nPeaks);
    f0Cands = (double*)malloc(sizeof(double)*F0NUMPEAK*F0CANDPERPEAK);
    
    maxnpeaks = min(maxnpeaks, nPeaks);                        //maximum number of peaks to use
    
    if (maxnpeaks>F0NUMPEAK)
    {
        for(ii=0;ii<nPeaks;ii++)
        {
            pfreq[ii] = pfreq1[ii];
			pmag_temp[ii] = pmag[ii];
        }
        
        minValArg(pfreq, nPeaks, &min_val, &min_ind);
        if (min_val==0)
        {
            pmag_temp[min_ind]=MAGSILENCE;
        }
        
        for (ii=0;ii<F0NUMPEAK;ii++)
        {
            maxValArg(pmag_temp, nPeaks, &max_val, &max_ind);
            f0Cands[oneTOtwo(ii,0,F0CANDPERPEAK)] = pfreq[max_ind];
            pmag_temp[max_ind] = MAGSILENCE;
            for(jj=0;jj<F0CANDPERPEAK;jj++)
            {
                f0Cands[oneTOtwo(ii,jj,F0CANDPERPEAK)] = f0Cands[oneTOtwo(ii,0,F0CANDPERPEAK)]/(float)(jj+1);
                
            }
        }
        
        f0CandCnt=0;
        for(ii=0;ii<F0NUMPEAK;ii++)
        {
            for(jj=0;jj<F0CANDPERPEAK;jj++)
            {
                f0down = f0Cands[oneTOtwo(ii,jj,F0CANDPERPEAK)];
                if((f0down<maxf0)&&(f0down>minf0))
                {
                    f0Cands[f0CandCnt] = f0down;
                    f0CandCnt++;
                }
                //removing all the close repetitions of the candidates, preferring the ones that are division of higher frequency peak
 
            }
        }
        
        if (f0CandCnt ==0)
        {
            f0 = 0 ;
            f0error = 100;
            return f0;
        }
        else
        {
            f0error=0;
        }
        
        TWM_C(pfreq, pmag, nPeaks, maxnpeaks, f0Cands, f0CandCnt, &f0, &f0error);
        
    }

    if((f0>0)&&(f0error>ef0max))
    {
        f0=0;
    }

    free(pfreq);
    free(pmag_temp);
    free(f0Cands);
    
    return f0;
}

/*This functino computed error for nPeaks number of peaks in peakMTX1 to peakMTX2*/
void computeTWMError(double **peakMTX1, int nCols1, double **peakMTX2, int nCols2, int maxnpeaks, double * pmag, double *f0Error, int nF0Cands, int PMorMP)
{
    int ii,jj, min_ind;
    double Ponddif, FreqDistance, MagFactor;
    
    if (PMorMP)
    {
        
        for(ii=0;ii<nF0Cands;ii++)
        {
            for(jj=0;jj<maxnpeaks;jj++)
            {
                min_ind = nearestElement(peakMTX1[ii][jj], peakMTX2[ii], nCols2, &FreqDistance);
                Ponddif = FreqDistance*pow(peakMTX1[ii][jj],-TWM_p);
                MagFactor = pmag[min_ind];
                //MagFactor = pmag[jj];
                f0Error[ii] = f0Error[ii] + Ponddif + MagFactor*(TWM_q*Ponddif-TWM_r);
                
            }
        }
        
    }
    else
    {
        for(ii=0;ii<nF0Cands;ii++)
        {
            for(jj=0;jj<maxnpeaks;jj++)
            {
                min_ind = nearestElement(peakMTX1[ii][jj], peakMTX2[ii], nCols2, &FreqDistance);
                Ponddif = FreqDistance*pow(peakMTX1[ii][jj],-TWM_p);
                MagFactor = pmag[jj];
                f0Error[ii] = f0Error[ii] + MagFactor*(Ponddif + MagFactor*(TWM_q*Ponddif-TWM_r));
                
            }
        }
    }
    
}

int nearestElement(double val, double *data, int len, double *min_val)
{
    int ii=0, min_ind=0;
    double diff1 = FLT_MAX, diff2;
    for(ii=0;ii<len;ii++)
    {
        diff2 = fabs(data[ii]-val);
        if(diff2 <diff1)
        {
            diff1 = diff2;
            min_ind = ii;
        }
    }
    *min_val = diff1;
    return min_ind;
}

int TWM_C(double *pfreq, double *pmag, int nPeaks, int maxnpeaks, double *f0c, int nf0c, double *f0, double *f0error)
{
    double Amax, min_val, maxMeasuredFreq, minF0Candidate, *ErrorPM, *ErrorMP, *pmag_local, **measMTX, **predMTX;
    int max_ind, min_ind,ii,jj, PMorMP, canMTXLen;
    
    maxValArg(pmag, nPeaks, &Amax, &max_ind);
    
    //computing the maximum number of harmonic index that we need to generate predicted peaks so that they are enough close to measured peaks
    //this will be ceil(maxMeasuredFreq/minFoCandidateFreq)
    minValArg(f0c, nf0c, &minF0Candidate, &min_ind);
    maxValArg(pfreq, nPeaks, &maxMeasuredFreq, &max_ind);
    canMTXLen = max(maxnpeaks, ceil(maxMeasuredFreq/minF0Candidate));
    
    predMTX = (double **)malloc(sizeof(double*)*nf0c);
    measMTX = (double **)malloc(sizeof(double*)*nf0c);
    
    for(ii=0;ii<nf0c;ii++)
    {
        predMTX[ii] = (double*)malloc(sizeof(double)*canMTXLen);
        for(jj=0;jj<canMTXLen;jj++)
        {
            predMTX[ii][jj] = (jj+1)*f0c[ii];
        }
        
        measMTX[ii] = (double *)malloc(sizeof(double)*nPeaks);
        for(jj=0;jj<nPeaks;jj++)
        {
            measMTX[ii][jj] = pfreq[jj];
        }
        
    }
    
    ErrorPM = (double*)malloc(sizeof(double)*nf0c);
    memset(ErrorPM, 0, sizeof(double)*nf0c);
    ErrorMP = (double*)malloc(sizeof(double)*nf0c);
    memset(ErrorMP, 0, sizeof(double)*nf0c);
    pmag_local = (double*)malloc(sizeof(double)*nPeaks);
    
    for(ii=0;ii<nPeaks;ii++)
    {
        pmag_local[ii] = pow(10,(pmag[ii]-Amax)/20.0);
    }
    
    maxnpeaks = min(maxnpeaks, nPeaks);
    PMorMP = 1;
    computeTWMError(predMTX, canMTXLen, measMTX, nPeaks, maxnpeaks, pmag_local, ErrorPM, nf0c,  PMorMP);
    PMorMP = 0;
    computeTWMError(measMTX, nPeaks, predMTX, canMTXLen, maxnpeaks, pmag_local, ErrorMP, nf0c,  PMorMP);
    
    //reusing ErrorPM as Error (total)
    for(ii=0;ii<nf0c;ii++)
    {
        ErrorPM[ii]= (ErrorPM[ii] + TWM_rho*ErrorMP[ii])/maxnpeaks ; 
    }
    
    minValArg(ErrorPM, nf0c, &min_val, &min_ind);
    *f0error = min_val;
    *f0 = f0c[min_ind];
    
    free(ErrorPM);
    free(ErrorMP);
    free(pmag_local);
    
    for(ii=0;ii<nf0c;ii++)
    {
        free(predMTX[ii]);
        free(measMTX[ii]);
    }
    free(predMTX);
    free(measMTX);
    
    
    return 1;
    
    
}


