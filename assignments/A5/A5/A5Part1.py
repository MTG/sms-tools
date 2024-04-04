import numpy as np
from scipy.signal import get_window
import math
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import dftModel as DFT
import utilFunctions as UF

""" 
A5-Part-1: Minimizing the frequency estimation error of a sinusoid

Write a function that estimates the frequency of a sinusoidal signal at a given time instant. The 
function should return the estimated frequency in Hz, together with the window size and the FFT 
size used in the analysis.  

The input arguments to the function are the wav file name including the path (inputFile) containing 
the sinusoidal signal, and the frequency of the sinusoid in Hz (f). The frequency of the input sinusoid  
can range between 100Hz and 2000Hz. The function should return a three element tuple of the estimated 
frequency of the sinusoid (fEst), the window size (M) and the FFT size (N) used.

The input wav file is a stationary audio signal consisting of a single sinusoid of length >=1 second. 
Since the signal is stationary you can just perform the analysis in a single frame, for example in 
the middle of the sound file (time equal to .5 seconds). The analysis process would be to first select 
a fragment of the signal equal to the window size, M, centered at .5 seconds, then compute the DFT 
using the dftAnal function, and finally use the peakDetection and peakInterp functions to obtain the 
frequency value of the sinusoid.

Use a Blackman window for analysis and a magnitude threshold t = -40 dB for peak picking. The window
size and FFT size should be chosen such that the difference between the true frequency (f) and the 
estimated frequency (fEst) is less than 0.05 Hz for the entire allowed frequency range of the input 
sinusoid. The window size should be the minimum positive integer of the form 100*k + 1 (where k is a 
positive integer) for which the frequency estimation error is < 0.05 Hz. For a window size M, take the
FFT size (N) to be the smallest power of 2 larger than M. 

HINT: Computing M theoritically using a formula might be complex in such cases. Instead, you need to 
follow a heuristic approach to determine the optimal value of M and N for a particular f. You can iterate
over all allowed values of window size M and stop when the condition is satisfied (i.e. the frequency
estimation error < 0.05 Hz).

Test case 1: If you run your code with inputFile = '../../sounds/sine-490.wav', f = 490.0 Hz, the optimal
values are M = 1101, N = 2048, fEst = 489.963 and the freqency estimation error is 0.037.

Test case 2: If you run your code with inputFile = '../../sounds/sine-1000.wav', f = 1000.0 Hz, the optimal
values are M = 1101, N = 2048, fEst = 1000.02 and the freqency estimation error is 0.02.

Test case 3: If you run your code with inputFile = '../../sounds/sine-200.wav', f = 200.0 Hz, the optimal
values are M = 1201, N = 2048, fEst = 200.038 and the freqency estimation error is 0.038.
"""
def minFreqEstErr(inputFile, f):
    """
    Inputs:
            inputFile (string) = wav file including the path
            f (float) = frequency of the sinusoid present in the input audio signal (Hz)
    Output:
            fEst (float) = Estimated frequency of the sinusoid (Hz)
            M (int) = Window size
            N (int) = FFT size
    """
    # analysis parameters:
    window = 'blackman'
    t = -40
    
    ### Your code here