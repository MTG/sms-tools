import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import harmonicModel as HM
import dftModel as DFT
import stft

eps = np.finfo(float).eps

"""
A6Part4 - Improving the implementation of the two way mismatch f0 estimation algorithm

Improve the performance of the current implementation of the two way mismatch algorithm in sms-tools 
used for fundamental frequency estimation. This is an optional open question and will not contribute 
towards the final grade. There is no definite answer for this question. Its main purpose is to 
understand the limitations of the current implementations of the TWM algorithm and to come up with 
some community driven solutions based on collective thinking. 

In this question you will directly modify the core functions that implement the TWM algorithm in 
sms-tools. To assist you with this task, we have copied all the needed functions into this python 
file. Hence, you just need to modify the functions in this file and not anywhere else.

Estimating fundamental frequency from an audio signal is still a challenging and unsolved problem 
to a large extent. By this time you might have also realized that many times the performance of the 
TWM f0 estimation algorithm falls short of the expectations. There can be a systematic explanation 
for the scenarios where TWM fails for specific categories or characteristics of the sounds. Some of 
the known scenarios where the current implementation of the TWM algorithm fails to estimate a correct 
fundamental frequency are:

1) Missing fundamental frequency: For many sounds the fundamental frequency component is very low and 
therefore during the spectral peak picking step we do not obtain any peak corresponding to the f0. 
Since the TWM algorithm implemented in sms-tools considers only the detected spectral peaks as the 
f0 candidates, we do not get any candidate corresponding to the f0. This causes f0 estimation to fail. 
For example, such a scenario is encountered in low pitched vocal sounds.

2) Pseudo-harmonicity in the sound. Many instruments such as piano exhibit some deviation from perfect 
harmonicity wherein their harmonic partials are not perfectly located at integral multiples of the 
fundamental frequency. Since the TWM algorithm computes error function assuming that the harmonic 
locations are at integral multiples, its performance is poorer when such deviations exist.

In this question we propose to work on these two scenarios. Go to freesound and download sound examples 
of low pitched vocal sounds and of piano. Run current implementation of TMW to identify the limitations 
and propose improvements to the code in order to obtain better f0 estimation for those two particular 
scenarios. 

The core TWM algorithm is implemented in the function TWM_p(), which takes in an array of f0 candidates 
and detect the candidate that has the lowest error. TWM_p() is called by f0Twm(), which generates 
f0 candidates (f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0]). This function also implements 
a memory based prunning of the f0 candidates. If the f0 contour is found to be stable (no drastic 
transitions across frames) then only the f0 candidates close to the stable f0 value are retained. 
f0Twm() is called for every audio frame by f0Detection().

You can use computeAndPlotF0(), which calls f0Detection() for estimating f0 for every audio frame. 
In addition, it also plots the f0 contour on the top of the spectrogram. If you set plot=1, it shows 
the plot, plot=2 saves the plot as can be seen in the code. 

Once you implement your proposed enhancement, discuss and share your ideas on the discussion forum 
assigned for A6Part4 - https://class.coursera.org/audio-001/forum/list?forum_id=10026. Along with the 
text you should include 2 plots showing the f0 contour before and after your changes. Use the same 
values of the analysis parameters while showing the improvement in the performance. in the discussion, 
also include a link to the sound in freesound. 

TIP: An identified limitation of the current implementation for the case of low vocal sounds is that 
it can only find f0 if there is a peak present in the magnitude spectrum. A possible improvement is 
to generate additional f0 candidates from the identified peaks. Another identified limitation for 
the case of piano sounds is the assumption of perfect harmonicity. For these sounds you can think 
of modifying the generation of the ideal harmonic series that is computed in the code, incorporating 
the typical deviation from harmonicity encountered in piano sounds.

NOTE: Before you start making changes in the TWM implementation make sure you have reached the best 
possible performance that can be achieved by tuning the analysis parameters. If the analysis parameters 
are inappropriately set, it is not completely meaningful to just improve the TWM implementation.

To maintain the integrity if the sms-tools package for future assignments, please make changes only 
to the functions in this file and not the other files in sms-tools.
"""

def computeAndPlotF0(inputFile = '../../sounds/piano.wav'):
    """
    Function to estimate fundamental frequency (f0) in an audio signal using TWM.
    Input:
        inputFile (string): wav file including the path    
    """
    window='hamming'
    M=2048
    N=2048
    H=256
    f0et=5.0
    t=-80
    minf0=100
    maxf0=300

    fs, x = UF.wavread(inputFile)                               #reading inputFile
    w  = get_window(window, M)                                  #obtaining analysis window    
    f0 = f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)  #estimating F0

    ## Code for plotting the f0 contour on top of the spectrogram
    # frequency range to plot
    maxplotfreq = 500.0    
    fontSize = 16
    plot = 1

    fig = plt.figure()
    ax = fig.add_subplot(111)

    mX, pX = stft.stftAnal(x, w, N, H)                      #using same params as used for analysis
    mX = np.transpose(mX[:,:int(N*(maxplotfreq/fs))+1])
    
    timeStamps = np.arange(mX.shape[1])*H/float(fs)                             
    binFreqs = np.arange(mX.shape[0])*fs/float(N)
    
    plt.pcolormesh(timeStamps, binFreqs, mX)
    plt.plot(timeStamps, f0, color = 'k', linewidth=1.5)
    
    plt.autoscale(tight=True)
    plt.ylabel('Frequency (Hz)', fontsize = fontSize)
    plt.xlabel('Time (s)', fontsize = fontSize)
    plt.legend(('f0',))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    ax.set_aspect((xLim[1]-xLim[0])/(2.0*(yLim[1]-yLim[0])))    

    if plot == 1: 
        plt.autoscale(tight=True) 
        plt.show()
    elif plot == 2:                   #you can save the plot too!
        fig.tight_layout()
        fig.savefig('f0_over_Spectrogram.png', dpi=150, bbox_inches='tight')



def f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et):
    """
    Fundamental frequency detection of a sound using twm algorithm
    x: input sound; fs: sampling rate; w: analysis window; 
    N: FFT size; t: threshold in negative dB, 
    minf0: minimum f0 frequency in Hz, maxf0: maximim f0 frequency in Hz, 
    f0et: error threshold in the f0 detection (ex: 5),
    returns f0: fundamental frequency
    """
    if (minf0 < 0):                                            # raise exception if minf0 is smaller than 0
        raise ValueError("Minumum fundamental frequency (minf0) smaller than 0")
    
    if (maxf0 >= 10000):                                       # raise exception if maxf0 is bigger than fs/2
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")
    
    if (H <= 0):                                               # raise error if hop size 0 or negative
        raise ValueError("Hop size (H) smaller or equal to 0")
        
    hN = N/2                                                   # size of positive spectrum
    hM1 = int(math.floor((w.size+1)/2))                        # half analysis window size by rounding
    hM2 = int(math.floor(w.size/2))                            # half analysis window size by floor
    x = np.append(np.zeros(hM2),x)                             # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(hM1))                             # add zeros at the end to analyze last sample
    pin = hM1                                                  # init sound pointer in middle of anal window          
    pend = x.size - hM1                                        # last sample to start a frame
    fftbuffer = np.zeros(N)                                    # initialize buffer for FFT
    w = w / sum(w)                                             # normalize analysis window
    f0 = []                                                    # initialize f0 output
    f0t = 0                                                    # initialize f0 track
    f0stable = 0                                               # initialize f0 stable
    while pin<pend:             
        x1 = x[pin-hM1:pin+hM2]                                  # select frame
        mX, pX = DFT.dftAnal(x1, w, N)                           # compute dft           
        ploc = UF.peakDetection(mX, t)                           # detect peak locations   
        iploc, ipmag, ipphase = UF.peakInterp(mX, pX, ploc)      # refine peak values
        ipfreq = fs * iploc/N                                    # convert locations to Hez
        f0t = f0Twm(ipfreq, ipmag, f0et, minf0, maxf0, f0stable)  # find f0
        if ((f0stable==0)&(f0t>0)) \
                or ((f0stable>0)&(np.abs(f0stable-f0t)<f0stable/5.0)):
            f0stable = f0t                                         # consider a stable f0 if it is close to the previous one
        else:
            f0stable = 0
        f0 = np.append(f0, f0t)                                  # add f0 to output array
        pin += H                                                 # advance sound pointer
    return f0

def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
    """
    Function that wraps the f0 detection function TWM, selecting the possible f0 candidates
    and calling the function TWM with them
    pfreq, pmag: peak frequencies and magnitudes, 
    ef0max: maximum error allowed, minf0, maxf0: minimum  and maximum f0
    f0t: f0 of previous frame if stable
    returns f0: fundamental frequency in Hz
    """
    if (minf0 < 0):                                  # raise exception if minf0 is smaller than 0
        raise ValueError("Minumum fundamental frequency (minf0) smaller than 0")
    
    if (maxf0 >= 10000):                             # raise exception if maxf0 is bigger than 10000Hz
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")
        
    if (pfreq.size < 3) & (f0t == 0):                # return 0 if less than 3 peaks and not previous f0
        return 0
    
    f0c = np.argwhere((pfreq>minf0) & (pfreq<maxf0))[:,0] # use only peaks within given range
    if (f0c.size == 0):                              # return 0 if no peaks within range
        return 0
    f0cf = pfreq[f0c]                                # frequencies of peak candidates
    f0cm = pmag[f0c]                                 # magnitude of peak candidates

    if f0t>0:                                        # if stable f0 in previous frame 
        shortlist = np.argwhere(np.abs(f0cf-f0t)<f0t/2.0)[:,0]   # use only peaks close to it
        maxc = np.argmax(f0cm)
        maxcfd = f0cf[maxc]%f0t
        if maxcfd > f0t/2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd>(f0t/4)): # or the maximum magnitude peak is not a harmonic
            shortlist = np.append(maxc, shortlist)
        f0cf = f0cf[shortlist]                         # frequencies of candidates                     

    if (f0cf.size == 0):                             # return 0 if no peak candidates
        return 0

    f0, f0error = TWM_p(pfreq, pmag, f0cf)        # call the TWM function with peak candidates
    
    if (f0>0) and (f0error<ef0max):                  # accept and return f0 if below max error allowed
        return f0
    else:
        return 0


def TWM_p(pfreq, pmag, f0c):
    """
    Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
    [better to use the C version of this function: UF_C.twm]
    pfreq, pmag: peak frequencies in Hz and magnitudes, 
    f0c: frequencies of f0 candidates
    returns f0, f0Error: fundamental frequency detected and its error
    """

    p = 0.5                                          # weighting by frequency value
    q = 1.4                                          # weighting related to magnitude of peaks
    r = 0.5                                          # scaling related to magnitude of peaks
    rho = 0.33                                       # weighting of MP error
    Amax = max(pmag)                                 # maximum peak magnitude
    maxnpeaks = 10                                   # maximum number of peaks used
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)                # initialize PM errors
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM) :                      # predicted to measured mismatch error
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1))*pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)    # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T)**(-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor*(q*Ponddif-r)).T
        harmonic = harmonic+f0c

    ErrorMP = np.zeros(harmonic.size)                # initialize MP errors
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size) :                    # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP]/f0c[i])
        nharm = (nharm>=1)*nharm + (nharm<1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm*f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP]**(-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10**((PeakMag-Amax)/20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor*(q*Ponddif-r)))

    Error = (ErrorPM[0]/MaxNPM) + (rho*ErrorMP/MaxNMP)  # total error
    f0index = np.argmin(Error)                       # get the smallest error
    f0 = f0c[f0index]                                # f0 with the smallest error

    return f0, Error[f0index]        
