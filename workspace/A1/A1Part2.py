import sys
import os
sys.path.append('../../software/models/')
from utilFunctions import wavread

"""
A1-Part-2: Basic operations with audio

Write a function that reads an audio file and returns the minimum and the maximum values of the audio
samples in that file.

The input to the function is the wav file name (including the path) and the output should be two floating
point values.

If you run your code using oboe-A4.wav as the input, the function should return the following output:
(-0.83486432, 0.56501967)
"""
def minMaxAudio(inputFile):
    """
    Input:
        inputFile: file name of the wav file (including path)
    Output:
        A tuple of the minimum and the maximum value of the audio samples, like: (min_val, max_val)
    """

    freq, wavArray = wavread(inputFile)
    min_val = min(wavArray)
    max_val = max(wavArray)
    
    return (min_val, max_val)
