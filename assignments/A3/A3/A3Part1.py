from scipy.fftpack import fft
import numpy as np
from fractions import gcd

"""
A3-Part-1: Minimize energy spread in DFT of sinusoids
Given a signal consisting of two sinusoids, write a function that selects the first M samples from 
the signal and returns the positive half of the DFT magnitude spectrum (in dB), such that it has 
only two non-zero values. 

M is to be calculated as the smallest positive integer for which the positive half of the DFT magnitude 
spectrum has only two non-zero values. To get the positive half of the spectrum, first compute the 
M point DFT of the input signal (for this you can use the fft function of scipy.fftpack, which is 
already imported in this script). Consider only the first (M/2)+1 samples of the DFT and compute the
magnitude spectrum of the positive half (in dB) as mX = 20*log10(abs(X[:M/2+1])), where X is the DFT 
of the input.

The input arguments to this function are the input signal x (of length W >= M) consisting of two 
sinusoids of frequency f1 and f2, the sampling frequency fs and the value of frequencies f1 and f2. 
The function should return the positive half of the magnitude spectrum mX. For this question, 
you can assume the input frequencies f1 and f2 to be positive integers and factors of fs, and 
that M is even. 

Due to the precision of the FFT computation, the zero values of the DFT are not zero but very small
values < 1e-12 (or -240 dB) in magnitude. For practical purposes, all values with absolute value less 
than 1e-6 (or -120 dB) can be considered to be zero. 

HINT: The DFT magnitude spectrum of a sinusoid has only one non-zero value (in the positive half of 
the DFT spectrum) when its frequency coincides with one of the DFT bin frequencies. This happens when 
the DFT size (M in this question) contains exactly an integer number of periods of the sinusoid. 
Since the signal in this question consists of two sinusoids, this condition should hold true for each 
of the sinusoids, so that the DFT magnitude spectrum has only two non-zero values, one per sinusoid. 

M can be computed as the Least Common Multiple (LCM) of the sinusoid periods (in samples). The LCM of
two numbers x, y can be computed as: x*y/GCD(x,y), where GCD denotes the greatest common divisor. In 
this script (see above) we have already imported fractions.gcd() function that computes the GCD. 

Test case 1: For an input signal x sampled at fs = 10000 Hz that consists of sinusoids of frequency 
f1 = 80 Hz and f2 = 200 Hz, you need to select M = 250 samples of the signal to meet the required 
condition. In this case, output mX is 126 samples in length and has non-zero values at bin indices 2 
and 5 (corresponding to the frequency values of 80 and 200 Hz, respectively). You can create a test 
signal x by generating and adding two sinusoids of the given frequencies.

Test case 2: For an input signal x sampled at fs = 48000 Hz that consists of sinusoids of frequency 
f1 = 300 Hz and f2 = 800 Hz, you need to select M = 480 samples of the signal to meet the required 
condition. In this case, output mX is 241 samples in length and has non-zero values at bin indices 3 
and 8 (corresponding to the frequency values of 300 and 800 Hz, respectively). You can create a test 
signal x by generating and adding two sinusoids of the given frequencies.
"""

def minimizeEnergySpreadDFT(x, fs, f1, f2):
    """
    Inputs:
        x (numpy array) = input signal 
        fs (float) = sampling frequency in Hz
        f1 (float) = frequency of the first sinusoid component in Hz
        f2 (float) = frequency of the second sinusoid component in Hz
    Output:
        The function should return 
        mX (numpy array) = The positive half of the DFT spectrum (in dB) of the M sample segment of x. 
                           mX is (M/2)+1 samples long (M is to be computed)
    """
    ## Your code here
