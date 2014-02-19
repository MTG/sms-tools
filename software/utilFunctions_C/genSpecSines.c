

#include "genSpecSines.h"
#include "tables.h"

void genbh92lobe_C(double *x, double *y, int N)
{
  int ii=0;
  
  for(ii=0;ii<N;ii++)
  {
    y[ii] = bh_92_1001[(int)(x[ii]*100) + BH_SIZE_BY2];
  }
  
  
  
}
  
void genspecsines_C(double *iploc, double *ipmag, double *ipphase, int n_peaks, double *real, double*imag, int size_spec)
{
	int ii=0,jj=0, ploc_int;
	int size_spec_half = (int)floor(size_spec/2);
	float bin_remainder,loc,mag;
	
	for(ii=0;ii<n_peaks; ii++)
	{
		loc = iploc[ii];
		bin_remainder = floor(loc + 0.5)-loc;
		ploc_int = (int)floor(loc+0.5);
		
		if((loc>=5)&&(loc<size_spec_half-4))
		{
			mag = pow(10,(ipmag[ii]/20.0));
			
			for(jj=-4;jj<5;jj++)
			{
				real[ploc_int+jj] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*cos(ipphase[ii]);
				imag[ploc_int+jj] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*sin(ipphase[ii]);
			}
		}
		else if((loc>0)&&(loc<5))
		{
			mag = pow(10,(ipmag[ii]/20.0));
			
			for(jj=-4;jj<5;jj++)
			{
				
				if(ploc_int+jj<0)
				{
					real[-1*(ploc_int+jj)] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*cos(ipphase[ii]);
					imag[-1*(ploc_int+jj)] += -1*mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*sin(ipphase[ii]);
					
				}
				else if (ploc_int+jj==0)
				{
					real[(ploc_int+jj)] += 2*mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*cos(ipphase[ii]);
				}
				else
				{
					real[(ploc_int+jj)] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*cos(ipphase[ii]);
					imag[ploc_int+jj] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*sin(ipphase[ii]);
				}
			}
		}
		else if((loc>=size_spec_half-4)&&(loc<size_spec_half-1))
		{
			mag = pow(10,(ipmag[ii]/20.0));
			
			for(jj=-4;jj<5;jj++)
			{
				if(ploc_int+jj>size_spec_half)
				{
					real[size_spec-(ploc_int+jj)] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*cos(ipphase[ii]);
					imag[size_spec-(ploc_int+jj)] += -1*mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*sin(ipphase[ii]);
					
				}
				else if(ploc_int+jj==size_spec_half)
				{
					real[(ploc_int+jj)] += 2*mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*cos(ipphase[ii]);
					
				}				
				else
				{
					real[(ploc_int+jj)] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*cos(ipphase[ii]);
					imag[ploc_int+jj] += mag*bh_92_1001[(int)((bin_remainder+jj)*100) + BH_SIZE_BY2]*sin(ipphase[ii]);
				}
			}
		}

	}
	
	for(ii=1;ii<size_spec_half;ii++)
	{
		real[size_spec_half+ii]=real[size_spec_half-ii];
		imag[size_spec_half+ii]=-1*imag[size_spec_half-ii];		
	}

}


