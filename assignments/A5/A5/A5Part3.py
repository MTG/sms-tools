import numpy as np
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF
import sineModel as SM
import matplotlib.pyplot as plt

"""
A5-Part-3: Tracking sinusoids of different amplitudes 

Perform a "good" sinusoidal analysis of a signal with two sinusoidal components of different amplitudes 
by specifying the parameters 'window type' and 'peak picking threshold' in the function mainlobeTracker() 
below. The function should return the parameters used, true and estimated tracks of frequency components, 
and their associated time stamps. 

We will consider a signal that has two sinusoidal components with a very large difference in their 
amplitude. We will use a synthetically generated signal with frequency components 440 Hz and 602 Hz, 
s = sin(2*pi*440*t) + 2e-3*sin(2*pi*602*t). As you see the amplitude difference is large. You will 
use the sound sines-440-602-hRange.wav. Listen to the sound and use sms-tools GUI or sonic visualizer 
to see its spectrogram. Notice the difference in the amplitudes of its components. 

You will not write any additional code in this question, but modify the parameters of the code to obtain 
the best possible results. There are three functions we have written for you. Please go through each 
function and understand it, but do not modify any of it.
1. mainlobeTracker(): This is the main function. Uses sineModel.py for sinusoidal analysis of the input 
sound. It takes an input audio file and uses the function sineModel.sineModelAnal(), tracks the mainlobes 
of the two sinusoids to obtain the two frequency tracks (fTrackEst) in the signal. It also computes the 
estimation error (meanErr) in frequency using the true frequency tracks obtained using genTrueFreqTracks().

mainlobeTracker() calls the following two functions:
2. genTimeStamps(): Generates the time stamps at which the sinuosid frequencies are estimated (one 
value per audio frame)
3. genTrueFreqTracks(): For the input sound sines-440-602-hRange.wav, the function generates the true 
frequency values, so that we can compare the true and the estimated frequency values. 

We will use sinusoidal Model to analyse this sound and extract the two components. We will use the 
sineModel.sineModelAnal() function for analysis. The code for analysis is already provided below with 
some parameters we have fixed. For analysis, we will use a window length (M) of 2047 samples, an FFT 
size (N) of 4096 samples and hop size (H) of 128 samples. For sine tracking, we set the minimum sine 
duration (minSineDur) to 0.02 seconds, freqDevOffset to 10 Hz and freqDevSlope to its default value of 
0.001. Since we need only two frequency component estimates at every frame, we set maxnSines = 2. 

Choose the parameters window and the peak picking threshold (t) such that the mean estimation error of 
each frequency components is less than 2 Hz. There is a range of values of M and t for which this is 
true and all of those values will be considered correct answers. You can plot the estimated and true 
frequency tracks to visualize the accuracy of estimation. The output is the set of parameters you used: 
window, t, the time stamps, estimated and the true frequency tracks. Note that choosing the wrong window 
might lead to tracking of one of the sidelobes of the high amplitude sinusoid instead of the mainlobe of 
the low amplitude sinusoid. 

We have written the function mainlobeTracker() and you have to edit the window and t values. For the window, choose 
one of 'boxcar', 'hanning', 'hamming', 'blackman', or 'blackmanharris'. t is specified in negative dB. These two 
parameters are marked as XX and you can edit the values as needed. 

As an example, choosing window = 'boxcar', t = -80.0, the mean estimation error is [0.142, 129.462] Hz. 
"""

def mainlobeTracker(inputFile = '../../sounds/sines-440-602-hRange.wav'):
    """
    Input:
           inputFile (string): wav file including the path
    Output:
           window (string): The window type used for analysis
           t (float) = peak picking threshold (negative dB)
           tStamps (numpy array) = A Kx1 numpy array of time stamps at which the frequency components were estimated
           fTrackEst = A Kx2 numpy array of estimated frequency values, one row per time frame, one column per component
           fTrackTrue = A Kx2 numpy array of true frequency values, one row per time frame, one column per component
    """       
    # Analysis parameters: Modify values of the parameters marked XX
    window = XX                                          # Window type
    t = XX                                               # threshold (negative dB)
    
    ### Go through the code below and understand it, do not modify anything ###   
    M = 2047                                             # Window size 
    N = 4096                                             # FFT Size
    H = 128                                              # Hop size in samples
    maxnSines = 2
    minSineDur = 0.02
    freqDevOffset = 10
    freqDevSlope = 0.001
    # read input sound
    fs, x = UF.wavread(inputFile)               
    w = get_window(window, M)                   # Compute analysis window
    tStamps = genTimeStamps(x.size, M, fs, H)   # Generate the tStamps to return
    # analyze the sound with the sinusoidal model
    fTrackEst, mTrackEst, pTrackEst = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)
    fTrackTrue = genTrueFreqTracks(tStamps)     # Generate the true frequency tracks
    tailF = 20                                 
    # Compute mean estimation error. 20 frames at the beginning and end not used to compute error
    meanErr = np.mean(np.abs(fTrackTrue[tailF:-tailF,:] - fTrackEst[tailF:-tailF,:]),axis=0)     
    print ("Mean estimation error = " + str(meanErr) + ' Hz')      # Print the error to terminal
    # Plot the estimated and true frequency tracks
    mX, pX = stft.stftAnal(x, w, N, H)
    maxplotfreq = 900.0
    binFreq = fs*np.arange(N*maxplotfreq/fs)/N
    plt.pcolormesh(tStamps, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]), cmap='hot_r')
    plt.plot(tStamps,fTrackTrue, 'o-', color = 'c', linewidth=3.0)
    plt.plot(tStamps,fTrackEst, color = 'y', linewidth=2.0)
    plt.legend(('True f1', 'True f2', 'Estimated f1', 'Estimated f2'))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.autoscale(tight=True)
    return window, float(t), tStamps, fTrackEst, fTrackTrue  # Output returned 

### Do not modify this function
def genTimeStamps(xlen, M, fs, H):
    # Generates the timeStamps as needed for output
    hM1 = int(np.floor((M+1)/2))                     
    hM2 = int(np.floor(M/2))                         
    xlen = xlen + 2*hM2
    pin = hM1
    pend = xlen - hM1                                     
    tStamps = np.arange(pin,pend,H)/float(fs)
    return tStamps

### Do not modify this function
def genTrueFreqTracks(tStamps):
    # Generates the true frequency values to compute estimation error
    # Specifically to sines-440-602-hRange.wav
    fTrack = np.zeros((len(tStamps),2))
    fTrack[:,0] = np.transpose(440*np.ones((len(tStamps),1)))
    fTrack[:,1] = np.transpose(602*np.ones((len(tStamps),1)))
    return fTrack
