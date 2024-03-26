import os
import sys
import numpy as np
import math
from scipy.signal import get_window
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import utilFunctions as UF
import harmonicModel as HM
import sineModel as SM
import stft

eps = np.finfo(float).eps

"""
A6Part1 - Estimate fundamental frequency in polyphonic audio signal

Set the analysis parameters used within the function estimateF0() to obtain a good estimate of the 
fundamental frequency (f0) corresponding to one melody within a complex audio signal. The signal 
is a cello recording cello-double-2.wav, in which two strings are played simultaneously. One string 
plays a constant drone while the other string plays a simple melody. You have to choose the analysis 
parameter values such that only the f0 frequency of the simple melody is tracked.

The input argument to the function is the wav file name including the path (inputFile). The function 
returns a numpy array of the f0 frequency values for each audio frame. For this question we take 
hopSize (H) = 256 samples. 

estimateF0() calls f0Detection() function of the harmonicModel.py, which uses the two way mismatch 
algorithm for f0 estimation. 

estimateF0() also plots the f0 contour on top of the spectrogram of the audio signal for you to 
visually analyse the performance of your chosen values for the analysis parameters. In this question 
we will only focus on the time segment between 0.5 and 4 seconds. So, your analysis parameter values 
should produce a good f0 contour in this time region.

In addition to plotting the f0 contour on the spectrogram, this function also synthesizes the f0 
contour. You can also evaluate the performance of your chosen analysis parameter values by listening 
to this synthesized wav file named 'synthF0Contour.wav'

Since there can be numerous combinations of the optimal analysis parameter values, the evaluation is 
done solely on the basis of the output f0 sequence. Note that only the segment of the f0 contour 
between time 0.5 to 4 seconds is used to evaluate the performance of f0 estimation.

Your assignment will be tested only on inputFile = '../../sounds/cello-double-2.wav'. So choose the 
analysis parameters using which the function estimates the f0 frequency contour corresponding to the 
string playing simple melody and not the drone. There is no separate test case for this question. 
You can keep working with the wav file mentioned above and when you think the performance is 
satisfactory you can submit the assignment. The plots can help you achieve a good performance. 

Be cautious while choosing the window size. Window size should be large enough to resolve the spectral 
peaks and small enough to preserve the note transitions. Very large window sizes may smear the f0 
contour at note transitions.

Depending on the parameters you choose and the capabilities of the hardware you use, the function 
might take a while to run (even half a minute in some cases). For this part of the assignment please 
refrain from posting your analysis parameters on the discussion forum. 
"""
def estimateF0(inputFile = '../../sounds/cello-double-2.wav'):
    """
    Function to estimate fundamental frequency (f0) in an audio signal. This function also plots the 
    f0 contour on the spectrogram and synthesize the f0 contour.
    Input:
        inputFile (string): wav file including the path
    Output:
        f0 (numpy array): array of the estimated fundamental frequency (f0) values
    """

    ### Change these analysis parameter values marked as XX
    window = XX
    M = XX
    N = XX
    f0et = XX
    t = XX
    minf0 = XX
    maxf0 = XX

    ### Do not modify the code below 
    H = 256                                                     #fix hop size
      
    fs, x = UF.wavread(inputFile)                               #reading inputFile
    w  = get_window(window, M)                                  #obtaining analysis window    
    
    ### Method 1
    f0 = HM.f0Detection(x, fs, w, N, H, t, minf0, maxf0, f0et)  #estimating F0
    startFrame = np.floor(0.5*fs/H)    
    endFrame = np.ceil(4.0*fs/H)
    f0[:startFrame] = 0
    f0[endFrame:] = 0
    y = UF.sinewaveSynth(f0, 0.8, H, fs)
    UF.wavwrite(y, fs, 'synthF0Contour.wav')

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
    plt.plot([0.5, 0.5], [0, maxplotfreq], color = 'b', linewidth=1.5)
    plt.plot([4.0, 4.0], [0, maxplotfreq], color = 'b', linewidth=1.5)
    
    
    plt.autoscale(tight=True)
    plt.ylabel('Frequency (Hz)', fontsize = fontSize)
    plt.xlabel('Time (s)', fontsize = fontSize)
    plt.legend(('f0',))
    
    xLim = ax.get_xlim()
    yLim = ax.get_ylim()
    ax.set_aspect((xLim[1]-xLim[0])/(2.0*(yLim[1]-yLim[0])))    

    if plot == 1: #save the plot too!
        plt.autoscale(tight=True) 
        plt.show()
    else:
        fig.tight_layout()
        fig.savefig('f0_over_Spectrogram.png', dpi=150, bbox_inches='tight')

    return f0
