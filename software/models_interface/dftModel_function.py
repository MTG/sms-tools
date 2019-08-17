# function to call the main analysis/synthesis functions in software/models/dftModel.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import get_window
import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../models/'))
import utilFunctions as UF
import dftModel as DFT

def main(inputFile = '../../sounds/piano.wav', window = 'blackman', M = 511, N = 1024, time = .2):
    """
    inputFile: input sound file (monophonic with sampling rate of 44100)
    window: analysis window type (choice of rectangular, hanning, hamming, blackman, blackmanharris)
    M: analysis window size (odd integer value)
    N: fft size (power of two, bigger or equal than than M)
    time: time  to start analysis (in seconds)
    """

    # read input sound (monophonic with sampling rate of 44100)
    fs, x = UF.wavread(inputFile)

    # compute analysis window
    w = get_window(window, M)

    # get a fragment of the input sound of size M
    sample = int(time*fs)
    if (sample+M >= x.size or sample < 0):                          # raise error if time outside of sound
        raise ValueError("Time outside sound boundaries")
    x1 = x[sample:sample+M]

    # compute the dft of the sound fragment
    mX, pX = DFT.dftAnal(x1, w, N)

    # compute the inverse dft of the spectrum
    y = DFT.dftSynth(mX, pX, w.size)*sum(w)

    # create figure
    plt.figure(figsize=(9, 6))

    # plot the sound fragment
    plt.subplot(4,1,1)
    plt.plot(time + np.arange(M)/float(fs), x1)
    plt.axis([time, time + M/float(fs), min(x1), max(x1)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')

    # plot the magnitude spectrum
    plt.subplot(4,1,2)
    plt.plot(float(fs)*np.arange(mX.size)/float(N), mX, 'r')
    plt.axis([0, fs/2.0, min(mX), max(mX)])
    plt.title ('magnitude spectrum: mX')
    plt.ylabel('amplitude (dB)')
    plt.xlabel('frequency (Hz)')
    # plot the phase spectrum
    plt.subplot(4,1,3)
    plt.plot(float(fs)*np.arange(pX.size)/float(N), pX, 'c')
    plt.axis([0, fs/2.0, min(pX), max(pX)])
    plt.title ('phase spectrum: pX')
    plt.ylabel('phase (radians)')
    plt.xlabel('frequency (Hz)')

    # plot the sound resulting from the inverse dft
    plt.subplot(4,1,4)
    plt.plot(time + np.arange(M)/float(fs), y)
    plt.axis([time, time + M/float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.tight_layout()
    plt.ion()
    plt.show()

if __name__ == "__main__":
    main()
