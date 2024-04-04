import numpy as np
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import sineModel as SM
import stft
import matplotlib.pyplot as plt

"""
A5-Part-4: Tracking sinusoids using the phase spectrum

Write a function selectFlatPhasePeak() that selects a sinusoid peak based on the flatness of the 
phase spectrum around the frequency of the sinusoid. The function will be used for tracking sinusoids 
in audio signals, as an alternate method to tracking the mainlobe peaks of the magnitude spectrum. 

In this question, you will implement an alternate way of tracking mainlobe of a sinusoid, using the 
phase spectrum. Recall that zero-phase windowing of sinusoid signal frame leads to a phase spectrum 
that is flat around the bins corresponding to frequency of the sinusoid. We will use this property 
of flatness of the phase spectrum as an alternative method to track the sinusoids. Note that this 
condition of flatness is satisfied only when the sinusoid is not time varying. For time-varying 
sinusoids, the condition fails. 

We will consider a signal that has two sinusoid components and has a transient in the middle of the 
audio file. You will use the sound sines-440-602-transient.wav. Listen to the sound and use sms-tools 
GUI or sonic visualizer to see its spectrogram. Notice the transient that occurs in the middle of the 
sound file, where tracking using phase is likely to fail. We also recommend you to use the sms-tools 
GUI and DFT model to plot the spectrum at different parts of the signal to see if you indeed observe 
that the phase spectrum is flat around the sinusoid frequencies. 

We will use sinusoidal model for analysis. We have modified the code in sineModel.sineModelAnal() to 
create a new function sineModelAnalEnhanced() which does a modified sine Tracking based on phase 
spectrum. Once we have the peaks estimated from the magnitude spectrum, we use a phase spectrum flatness 
measure around each peak location to select or reject the peak. 

You will implement the function selectFlatPhasePeak() that checks for the flatness of the phase spectrum 
around the peak location. Given the peak location (p), the positive half of the phase spectrum (pX) and 
a threshold (phaseDevThres), you will compute the standard deviation of 5 samples of pX around the peak 
location (two samples either side and the sample at p itself) and compare it with the threshold. Based 
on the comparison, return a boolean variable selectFlag, which is True if the standard deviation is less 
than the threshold (and hence the phase is flat), else False (phase is not flat). We will use a small 
phase deviation threshold of 0.01 radian. In short, selectFlatPhasePeak() that returns True if the 
standard deviation of five samples of the phase spectrum pX around the input index p is less than the 
given threshold, else False.

Read through the function sineModelAnalEnhanced() and understand it thoroughly before implementing 
selectFlatPhasePeak() function. The function sineModelAnalEnhanced() takes an input audio file and 
uses phase based sinusoid tracking to obtain the two frequency tracks (fTrackEst) in the signal. 
Since we need only two sinusoids every frame, we only consider the frames where we get two selected 
peaks, and ignore the other frames. You can plot the estimated and true frequency tracks to visualize 
the accuracy of estimation (code provided). 

Test case 1: With pX = np.array([1.0, 1.2, 1.3, 1.4, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8]), p = 3, and 
phaseDevThres = 0.25, the function selectFlatPhasePeak() returns selectFlag = True. 

Test case 2: With pX = np.array([1.0, 1.2, 1.3, 1.4, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8]), p = 3, and 
phaseDevThres = 0.1, the function selectFlatPhasePeak() returns selectFlag = False.

Test case 3: With pX = np.array([2.39, 2.40, 2.40, 2.41, 3.37, 2.45, 2.46, 2.46, 2.29, 1.85, 2.34, 
2.18, 2.93, 2.10, 3.39, 2.41, 2.41, 2.40, 2.40, 2.40, 1.46, 0.23, 0.98, 0.41, 0.37, 0.40, 0.41, 
0.87, 0.51, 0.67]), p = 17, and phaseDevThres = 0.01, the function selectFlatPhasePeak() 
returns selectFlag = True

As an example, when you run sineModelAnalEnhanced(inputFile= './sines-440-602-transient.wav'), if 
you have implemented selectFlatPhasePeak() function correctly, you will see two sinusoid tracks in 
the beginning and end of the audio file, while there are no tracks in the middle of the audio file. 
This is due to the transients present in the middle of the audio file, where phase based tracking of 
sinusoids fails. 

"""

## Complete this function
def selectFlatPhasePeak(pX, p, phaseDevThres):
    """
    Function to select a peak index based on phase flatness measure. 
    Input: 
            pX (numpy array) = The phase spectrum of the frame
            p (positive integer) = The index of peak in the magnitude spectrum
            phaseDevThres (float) = The threshold value to measure flatness of phase
    Output: 
            selectFlag (Boolean) = True, if the peak at index p is a mainlobe, False otherwise
    """
    #Your code here
    

### Go through the code below and understand it, but do not modify anything ###
def sineModelAnalEnhanced(inputFile= '../../sounds/sines-440-602-transient.wav'):
    """
    Input:
           inputFile (string): wav file including the path
    Output:
           tStamps: A Kx1 numpy array of time stamps at which the frequency components were estimated
           tfreq: A Kx2 numpy array of frequency values, one column per component
    """
    phaseDevThres = 1e-2                                   # Allowed deviation in phase
    M = 2047                                               # window size
    N = 4096                                               # FFT size 
    t = -80                                                # threshold in negative dB
    H = 128                                                # hop-size
    window='blackman'                                      # window type
    fs, x = UF.wavread(inputFile)                          # Read input file
    w = get_window(window, M)                              # Get the window
    hM1 = int(np.floor((w.size+1)/2))                      # half analysis window size by rounding
    hM2 = int(np.floor(w.size/2))                          # half analysis window size by floor
    x = np.append(np.zeros(hM2),x)                         # add zeros at beginning to center first window at sample 0
    x = np.append(x,np.zeros(hM2))                         # add zeros at the end to analyze last sample
    pin = hM1                                              # initialize sound pointer in middle of analysis window       
    pend = x.size - hM1                                    # last sample to start a frame
    tStamps = np.arange(pin,pend,H)/float(fs)              # Generate time stamps
    w = w / sum(w)                                         # normalize analysis window
    tfreq = np.array([])
    while pin<pend:                                        # while input sound pointer is within sound            
        x1 = x[pin-hM1:pin+hM2]                            # select frame
        mX, pX = SM.DFT.dftAnal(x1, w, N)                  # compute dft
        ploc = UF.peakDetection(mX, t)                     # detect locations of peaks
        ###### CODE DIFFERENT FROM sineModelAnal() #########
        # Phase based mainlobe tracking
        plocSelMask = np.zeros(len(ploc))                  
        for pindex, p in enumerate(ploc):
            if p > 2 and p < (len(pX) - 2):                    # Peaks at either end of the spectrum are not processed
                if selectFlatPhasePeak(pX, p, phaseDevThres):  # Select the peak if the phase spectrum around the peak is flat
                    plocSelMask[pindex] = 1        
            else:
                plocSelMask[pindex] = 1                        
        plocSel = ploc[plocSelMask.nonzero()[0]]               # Select the ones chosen
        if len(plocSel) != 2:                                  # Ignoring frames that don't return two selected peaks
            ipfreq = [0.0, 0.0]
        else:
            iploc, ipmag, ipphase = UF.peakInterp(mX, pX, plocSel) # Only selected peaks to refine peak values by interpolation
            ipfreq = fs*iploc/float(N)                             # convert peak locations to Hertz
        ###### CODE DIFFERENT FROM sineModelAnal() #########
        if pin == hM1:                                        # if first frame initialize output frequency track
            tfreq = ipfreq 
        else:                                                 # rest of frames append values to frequency track
            tfreq = np.vstack((tfreq, ipfreq))
        pin += H
    # Plot the estimated frequency tracks
    mX, pX = stft.stftAnal(x, w, N, H)
    maxplotfreq = 1500.0
    binFreq = fs*np.arange(N*maxplotfreq/fs)/N
    numFrames = int(mX[:,0].size)
    frmTime = H*np.arange(numFrames)/float(fs) 
    plt.pcolormesh(frmTime, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]), cmap='hot_r')
    plt.plot(tStamps,tfreq[:,0], color = 'y', linewidth=2.0)
    plt.plot(tStamps,tfreq[:,1], color = 'c', linewidth=2.0)
    plt.legend(('Estimated f1', 'Estimated f2'))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.autoscale(tight=True)
    return tStamps, tfreq
