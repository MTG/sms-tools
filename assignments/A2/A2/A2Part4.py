import numpy as np

"""
A2-Part-4: Implement the inverse discrete Fourier transform (IDFT)

Write a function that implements the inverse discrete Fourier transform (IDFT). Given a frequency 
spectrum X of length N, the function should return its IDFT x, also of length N. Assume that the 
frequency index of the input spectrum ranges from 0 to N-1.

The input argument to the function is a numpy array X of the frequency spectrum and the function should return 
a numpy array of the IDFT of X.

Remember to scale the output appropriately.

EXAMPLE: If you run your function using X = np.array([1 ,1 ,1 ,1]), the function should return the following numpy 
array x: array([  1.00000000e+00 +0.00000000e+00j,   -4.59242550e-17 +5.55111512e-17j,
    0.00000000e+00 +6.12323400e-17j,   8.22616137e-17 +8.32667268e-17j])

Notice that the output numpy array is essentially [1, 0, 0, 0]. Instead of exact 0 we get very small
numerical values of the order of 1e-15, which can be ignored. Also, these small numerical errors are 
machine dependent and might be different in your case.

In addition, an interesting test of the IDFT function can be done by providing the output of the DFT of 
a sequence as the input to the IDFT. See if you get back the original time domain sequence.

"""
def IDFT(X):
    """
    Input:
        X (numpy array) = frequency spectrum (length N)
    Output:
        The function should return a numpy array of length N 
        x (numpy array) = The N point IDFT of the frequency spectrum X
    """
    ## Your code here