import sys
import os
sys.path.append('../../software/models/')
from utilFunctions import wavread

"""
A1-Part-1: Reading an audio file

Write a function that reads an audio file and returns 10 consecutive samples of the file starting from 
the 50001th sample. This means that the output should exactly contain the 50001th sample to the 50010th 
sample (10 samples). 

The input to the function is the file name (including the path) and the output should be a numpy array 
containing 10 samples.

If you use the wavread function from the utilFunctions module the input samples will be automatically 
converted to floating point numbers with a range from -1 to 1, which is what we want. 

Remember that in python, the index of the first sample of an array is 0 and not 1.

If you run your code using piano.wav as the input, the function should return the following 10 samples:  
array([-0.06213569, -0.04541154, -0.02734458, -0.0093997 ,  0.00769066,	0.02319407,  0.03503525, 
0.04309214, 0.04626606,  0.0441908], dtype=float32)
"""

def readAudio(inputFile):
    """
    Input:
        inputFile: the path to the wav file      
    Output:
        The function should return a numpy array that contains 10 samples of the audio.
    """
    ## Your code here

