import numpy as np
from scipy.signal import get_window
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../software/models/'))
import stft
import utilFunctions as UF
import sineModel as SM
import matplotlib.pyplot as plt

"""
A5-Part-5: Sinusoidal modeling of a multicomponent signal (optional)

Perform a sinusoidal analysis of a complex synthetic signal, exploring the different parameters of 
the model. Use the sound multiSines.wav and post your comments on the forum thread specific for 
this topic. 

This is an open question without a single specific answer. We will use the sound multiSines.wav, which 
is a synthetic audio signal with sharp attacks, close frequency components with a wide range of 
amplitudes, and time varying chirps with frequency tracks that cross over. All these characteristics 
make this signal difficult to analyze with sineModel. Listen to the sound and use sms-tools GUI or 
sonic visualizer to see its spectrogram. 

We have written a basic code for sinusoidal analysis, you are free to modify the code. The function 
saves the output sound as <inputFileName>_sineModel.wav which you can listen to and visualize. 

You will get credit for this question by just attempting the question and submitting it. Share your 
thoughts on the forum thread (https://class.coursera.org/audio-002/forum/list?forum_id=10022) on the 
parameter choices you made, tradeoffs that you encountered, and quality of the reconstruction you achieved. 

"""
def exploreSineModel(inputFile='../../sounds/multiSines.wav'):
    """
    Input:
            inputFile (string) = wav file including the path
    Output: 
            return True
            Discuss on the forum!
    """
    window='hamming'                            # Window type
    M=3001                                      # Window size in sample
    N=4096                                      # FFT Size
    t=-80                                       # Threshold                
    minSineDur=0.02                             # minimum duration of a sinusoid
    maxnSines=15                                # Maximum number of sinusoids at any time frame
    freqDevOffset=10                            # minimum frequency deviation at 0Hz
    freqDevSlope=0.001                          # slope increase of minimum frequency deviation
    Ns = 512                                    # size of fft used in synthesis
    H = 128                                     # hop size (has to be 1/4 of Ns)
    
    fs, x = UF.wavread(inputFile)               # read input sound
    w = get_window(window, M)                   # compute analysis window

    # analyze the sound with the sinusoidal model
    tfreq, tmag, tphase = SM.sineModelAnal(x, fs, w, N, H, t, maxnSines, minSineDur, freqDevOffset, freqDevSlope)

    # synthesize the output sound from the sinusoidal representation
    y = SM.sineModelSynth(tfreq, tmag, tphase, Ns, H, fs)

    # output sound file name
    outputFile = os.path.basename(inputFile)[:-4] + '_sineModel.wav'

    # write the synthesized sound obtained from the sinusoidal synthesis
    UF.wavwrite(y, fs, outputFile)

    # create figure to show plots
    plt.figure(figsize=(12, 9))

    # frequency range to plot
    maxplotfreq = 5000.0

    # plot the input sound
    plt.subplot(3,1,1)
    plt.plot(np.arange(x.size)/float(fs), x)
    plt.axis([0, x.size/float(fs), min(x), max(x)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('input sound: x')
                
    # plot the sinusoidal frequencies
    plt.subplot(3,1,2)
    if (tfreq.shape[1] > 0):
        numFrames = tfreq.shape[0]
        frmTime = H*np.arange(numFrames)/float(fs)
        tfreq[tfreq<=0] = np.nan
        plt.plot(frmTime, tfreq)
        plt.axis([0, x.size/float(fs), 0, maxplotfreq])
        plt.title('frequencies of sinusoidal tracks')

    # plot the output sound
    plt.subplot(3,1,3)
    plt.plot(np.arange(y.size)/float(fs), y)
    plt.axis([0, y.size/float(fs), min(y), max(y)])
    plt.ylabel('amplitude')
    plt.xlabel('time (sec)')
    plt.title('output sound: y')

    plt.tight_layout()
    plt.show()
    return True