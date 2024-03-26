import numpy as np
import sys
sys.path.append('../../software/models/')
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import get_window
from dftModel import dftAnal
"""
A3-part-5: FFT size and zero-padding (Optional)

Write a function that takes in an input signal, computes three different FFTs on the input and returns 
the first 80 samples of the positive half of the FFT magnitude spectrum (in dB) in each case. 

This part is a walk-through example to provide some insights into the effects of the length of signal 
segment, the FFT size, and zero-padding on the FFT of a sinusoid. The input to the function is x, which
is 512 samples of a real sinusoid of frequency 110 Hz and the sampling frequency fs = 1000 Hz. You will 
first extract the first 256 samples of the input signal and store it as a separate variable xseg. You 
will then generate two 'hamming' windows w1 and w2 of size 256 and 512 samples, respectively (code given
below). The windows are used to smooth the input signal. Use dftAnal to obtain the positive half of the 
FFT magnitude spectrum (in dB) for the following cases:
Case-1: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 256
Case-2: Input signal x (512 samples), window w2 (512 samples), and FFT size of 512
Case-3: Input signal xseg (256 samples), window w1 (256 samples), and FFT size of 512 (Implicitly does a 
        zero-padding of xseg by 256 samples)
Return the first 80 samples of the positive half of the FFT magnitude spectrum output by dftAnal. 

To understand better, plot the output of dftAnal for each case on a common frequency axis. Let mX1, mX2, 
mX3 represent the outputs of dftAnal in each of the Cases 1, 2, and 3 respectively. You will see that 
mX3 is the interpolated version of mX1 (zero-padding leads to interpolation of the DFT). You will also 
observe that the 'mainlobe' of the magnitude spectrum in mX2 will be much smaller than that in mX1 and 
mX3. This shows that choosing a longer segment of signal for analysis leads to a narrower mainlobe with 
better frequency resolution and less spreading of the energy of the sinusoid. 

If we were to estimate the frequency of the sinusoid using its DFT, a first principles approach is to 
choose the frequency value of the bin corresponding to the maximum in the DFT magnitude spectrum. 
Some food for thought: if you were to take this approach, which of the Cases 1, 2, or 3 will give you 
a better estimate of the frequency of the sinusoid ? Comment and discuss on the forums!

Test case 1: The input signal is x (of length 512 samples), the output is a tuple with three elements: 
(mX1_80, mX2_80, mX3_80) where mX1_80, mX2_80, mX3_80 are the first 80 samples of the magnitude spectrum 
output by dftAnal in cases 1, 2, and 3, respectively. 

"""
def zpFFTsizeExpt(x, fs):
    """
    Inputs:
        x (numpy array) = input signal (2*M = 512 samples long)
        fs (float) = sampling frequency in Hz
    Output:
        The function should return a tuple (mX1_80, mX2_80, mX3_80)
        mX1_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-1
        mX2_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-2
        mX3_80 (numpy array): The first 80 samples of the magnitude spectrum output of dftAnal for Case-3
        
    The first few lines of the code to generate xseg and the windows have been written for you, 
    please use it and do not modify it. 
    """
    
    M = len(x)/2
    xseg = x[:M]
    w1 = get_window('hamming',M)
    w2 = get_window('hamming',2*M)
    ## Your code here 
