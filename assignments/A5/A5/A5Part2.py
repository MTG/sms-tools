import numpy as np
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF
import sineModel as SM
import matplotlib.pyplot as plt
""" 
A5-Part-2: Tracking a two component chirp 

Perform a "good" sinusoidal analysis of a two component chirp by specifying the parameter 
window-size in the function chirpTracker() below. The estimation and tracking of the two varying 
frequencies should result in a mean error smaller than 2Hz. The function returns the parameters 
used, true and estimated tracks of frequency components, and their associated time stamps. 

In this part, you will use the sound chirp-150-190-linear.wav, which is a linear chirp with two 
sinusoids of 150Hz and 190 Hz, whose frequency increases in two seconds to 1400Hz and 1440Hz 
respectively. Listen to the sound and use sms-tools GUI or sonic visualizer to see its spectrogram. 

You will not write any additional code in this question, but modify the parameter of the code to 
obtain the best possible results. There are three functions we have written for you. Please go through 
each function and understand it, but do not modify any of it.
1. chirpTracker(): This is the main function. Uses sineModel.py for sinusoidal analysis of the input 
sound. It takes an input audio file and uses the function sineModel.sineModelAnal() to obtain the two 
frequency tracks (fTrackEst) in the chirp and computes the estimation error (meanErr) using the true 
frequency tracks obtained using genTrueFreqTracks().

chirpTracker() calls the following two functions:
2. genTimeStamps(): Generates the time stamps at which the frequencies of sinusoids are estimated (one 
value per frame)
3. genTrueFreqTracks(): For the input sound chirp-150-190-linear.wav, the function generates the true 
frequency values, so that we can compare the true and the estimated frequency values. 

We will use the sinusoidal model to analyse this sound and extract the two components. We will use the 
sineModel.sineModelAnal() function for analysis. The code for analysis is already provided below 
with some parameters we have fixed. For analysis, we will use a blackman window. Since we need only 
two frequency component estimates every frame, we set maxnSines = 2. 

Choose the parameter window-size (M) such that the mean estimation error (meanErr) of each frequency 
component is less than 2 Hz. There is a range of values of M for which this is true and all of those 
values will be considered correct answers. You can plot the estimated and true frequency tracks to 
visualize the accuracy of estimation. 

We have written the function chirpTracker() for you and you just have to edit M. It is marked as XX 
and you can edit its value as needed. 

As an example, choosing M = 1023, the mean estimation error is [13.677,  518.409] Hz, 
which as you can see do not give us the desired estimation errors. 

"""
def chirpTracker(inputFile='../../sounds/chirp-150-190-linear.wav'):
    """
    Input:
           inputFile (string) = wav file including the path
    Output:
           M (int) = Window length
           H (int) = hop size in samples
           tStamps (numpy array) = A Kx1 numpy array of time stamps at which the frequency components were estimated
           fTrackEst (numpy array) = A Kx2 numpy array of estimated frequency values, one row per time frame, one column per component
           fTrackTrue (numpy array) = A Kx2 numpy array of true frequency values, one row per time frame, one column per component
           K is the number of frames
    """
    # Analysis parameters: Modify values of the parameters marked XX
    M = XX                                # Window size in samples
    
    ### Go through the code below and understand it, do not modify anything ###    
    H = 128                                     # Hop size in samples
    N = int(pow(2, np.ceil(np.log2(M))))        # FFT Size, power of 2 larger than M
    t = -80.0                                   # threshold
    window = 'blackman'                         # Window type
    maxnSines = 2                               # Maximum number of sinusoids at any time frame
    minSineDur = 0.0                            # minimum duration set to zero to not do tracking
    freqDevOffset = 30                          # minimum frequency deviation at 0Hz
    freqDevSlope = 0.001                        # slope increase of minimum frequency deviation
    
    fs, x = UF.wavread(inputFile)               # read input sound
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
    maxplotfreq = 1500.0
    binFreq = fs*np.arange(N*maxplotfreq/fs)/N
    plt.pcolormesh(tStamps, binFreq, np.transpose(mX[:,:N*maxplotfreq/fs+1]),cmap = 'hot_r')
    plt.plot(tStamps,fTrackTrue, 'o-', color = 'c', linewidth=3.0)
    plt.plot(tStamps,fTrackEst, color = 'y', linewidth=2.0)
    plt.legend(('True f1', 'True f2', 'Estimated f1', 'Estimated f2'))
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.autoscale(tight=True)
    plt.show()
    return M, H, tStamps, fTrackEst, fTrackTrue  # Output returned 

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
    # Specifically to chirp-150-190-linear.wav
    fTrack = np.zeros((len(tStamps),2))
    fTrack[:,0] = np.transpose(np.linspace(190, 190+1250, len(tStamps)))
    fTrack[:,1] = np.transpose(np.linspace(150, 150+1250, len(tStamps)))
    return fTrack
    
