import numpy as np

"""
A2-Part-2: Generate a complex sinusoid 

Write a function to generate the complex sinusoid that is used in DFT computation of length N (samples), 
corresponding to the frequency index k. Note that the complex sinusoid used in DFT computation has a 
negative sign in the exponential function.

The amplitude of such a complex sinusoid is 1, the length is N, and the frequency in radians is 2*pi*k/N.

The input arguments to the function are two positive integers, k and N, such that k < N-1. 
The function should return cSine, a numpy array of the complex sinusoid.

EXAMPLE: If you run your function using N=5 and k=1, the function should return the following numpy array cSine:
array([ 1.0 + 0.j,  0.30901699 - 0.95105652j, -0.80901699 - 0.58778525j, -0.80901699 + 0.58778525j, 
0.30901699 + 0.95105652j])
"""
def genComplexSine(k, N):
    """
    Inputs:
        k (integer) = frequency index of the complex sinusoid of the DFT
        N (integer) = length of complex sinusoid in samples
    Output:
        The function should return a numpy array
        cSine (numpy array) = The generated complex sinusoid (length N)
    """
    ## Your code here