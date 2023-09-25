import copy
import os
import subprocess
import sys

import numpy as np
from scipy.fft import fft, ifft, fftshift
from scipy.io.wavfile import write, read
from scipy.signal import resample
from scipy.signal.windows import blackmanharris, triang

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), './utilFunctions_C/'))
try:
    import utilFunctions_C as UF_C
except ImportError:
    print("\n")
    print("-------------------------------------------------------------------------------")
    print("Warning:")
    print("Cython modules for some of the core functions were not imported.")
    print("Please refer to the README.md file in the 'sms-tools' directory,")
    print("for the instructions to compile the cython modules.")
    print("Exiting the code!!")
    print("-------------------------------------------------------------------------------")
    print("\n")
    sys.exit(0)

winsound_imported = False
if sys.platform == "win32":
    try:
        import winsound

        winsound_imported = True
    except:
        print("You won't be able to play sounds, winsound could not be imported")


def isPower2(num):
    """
	Check if num is power of two
	"""
    return ((num & (num - 1)) == 0) and num > 0


INT16_FAC = (2 ** 15)
INT32_FAC = (2 ** 31)
INT64_FAC = (2 ** 63)
norm_fact = {'int16': INT16_FAC, 'int32': INT32_FAC, 'int64': INT64_FAC, 'float32': 1.0, 'float64': 1.0}


def wavread(filename):
    """
	Read a sound file and convert it to a normalized floating point array
	filename: name of file to read
	returns fs: sampling rate of file, x: floating point array
	"""

    if (os.path.isfile(filename) == False):  # raise error if wrong input file
        raise ValueError("Input file is wrong")

    fs, x = read(filename)

    if (len(x.shape) != 1):  # raise error if more than one channel
        raise ValueError("Audio file should be mono")

    if (fs != 44100):  # raise error if more than one channel
        raise ValueError("Sampling rate of input sound should be 44100")

    # scale down and convert audio into floating point number in range of -1 to 1
    x = np.float32(x) / norm_fact[x.dtype.name]
    return fs, x


def wavplay(filename):
    """
	Play a wav audio file from system using OS calls
	filename: name of file to read
	"""
    if (os.path.isfile(filename) == False):  # raise error if wrong input file
        print("Input file does not exist. Make sure you computed the analysis/synthesis")
    else:
        if sys.platform == "linux" or sys.platform == "linux2":
            # linux
            subprocess.call(["aplay", filename])

        elif sys.platform == "darwin":
            # OS X
            subprocess.call(["afplay", filename])
        elif sys.platform == "win32":
            if winsound_imported:
                winsound.PlaySound(filename, winsound.SND_FILENAME)
            else:
                print("Cannot play sound, winsound could not be imported")
        else:
            print("Platform not recognized")


def wavwrite(y, fs, filename):
    """
	Write a sound file from an array with the sound and the sampling rate
	y: floating point array of one dimension, fs: sampling rate
	filename: name of file to create
	"""

    x = copy.deepcopy(y)  # copy array
    x *= INT16_FAC  # scaling floating point -1 to 1 range signal to int16 range
    x = np.int16(x)  # converting to int16 type
    write(filename, fs, x)


def peakDetection(mX, t):
    """
	Detect spectral peak locations
	mX: magnitude spectrum, t: threshold
	returns ploc: peak locations
	"""

    thresh = np.where(np.greater(mX[1:-1], t), mX[1:-1], 0)  # locations above threshold
    next_minor = np.where(mX[1:-1] > mX[2:], mX[1:-1], 0)  # locations higher than the next one
    prev_minor = np.where(mX[1:-1] > mX[:-2], mX[1:-1], 0)  # locations higher than the previous one
    ploc = thresh * next_minor * prev_minor  # locations fulfilling the three criteria
    ploc = ploc.nonzero()[0] + 1  # add 1 to compensate for previous steps
    return ploc


def peakInterp(mX, pX, ploc):
    """
	Interpolate peak values using parabolic interpolation
	mX, pX: magnitude and phase spectrum, ploc: locations of peaks
	returns iploc, ipmag, ipphase: interpolated peak location, magnitude and phase values
	"""

    val = mX[ploc]  # magnitude of peak bin
    lval = mX[ploc - 1]  # magnitude of bin at left
    rval = mX[ploc + 1]  # magnitude of bin at right
    iploc = ploc + 0.5 * (lval - rval) / (lval - 2 * val + rval)  # center of parabola
    ipmag = val - 0.25 * (lval - rval) * (iploc - ploc)  # magnitude of peaks
    ipphase = np.interp(iploc, np.arange(0, pX.size), pX)  # phase of peaks by linear interpolation
    return iploc, ipmag, ipphase


def sinc(x, N):
    """
	Generate the main lobe of a sinc function (Dirichlet kernel)
	x: array of indexes to compute; N: size of FFT to simulate
	returns y: samples of the main lobe of a sinc function
	"""

    y = np.sin(N * x / 2) / np.sin(x / 2)  # compute the sinc function
    y[np.isnan(y)] = N  # avoid NaN if x == 0
    return y


def genBhLobe(x):
    """
	Generate the main lobe of a Blackman-Harris window
	x: bin positions to compute (real values)
	returns y: main lobe os spectrum of a Blackman-Harris window
	"""

    N = 512  # size of fft to use
    f = x * np.pi * 2 / N  # frequency sampling
    df = 2 * np.pi / N
    y = np.zeros(x.size)  # initialize window
    consts = [0.35875, 0.48829, 0.14128, 0.01168]  # window constants
    for m in range(0, 4):  # iterate over the four sincs to sum
        y += consts[m] / 2 * (sinc(f - df * m, N) + sinc(f + df * m, N))  # sum of scaled sinc functions
    y = y / N / consts[0]  # normalize
    return y


def genSpecSines(ipfreq, ipmag, ipphase, N, fs):
    """
	Generate a spectrum from a series of sine values, calling a C function
	ipfreq, ipmag, ipphase: sine peaks frequencies, magnitudes and phases
	N: size of the complex spectrum to generate; fs: sampling frequency
	returns Y: generated complex spectrum of sines
	"""

    Y = UF_C.genSpecSines(N * ipfreq / float(fs), ipmag, ipphase, N)
    return Y


def genSpecSines_p(ipfreq, ipmag, ipphase, N, fs):
    """
	Generate a spectrum from a series of sine values
	iploc, ipmag, ipphase: sine peaks locations, magnitudes and phases
	N: size of the complex spectrum to generate; fs: sampling rate
	returns Y: generated complex spectrum of sines
	"""

    Y = np.zeros(N, dtype=complex)  # initialize output complex spectrum
    hN = N // 2  # size of positive freq. spectrum
    for i in range(0, ipfreq.size):  # generate all sine spectral lobes
        loc = N * ipfreq[i] / fs  # it should be in range ]0,hN-1[
        if loc == 0 or loc > hN - 1: continue
        binremainder = round(loc) - loc
        lb = np.arange(binremainder - 4, binremainder + 5)  # main lobe (real value) bins to read
        lmag = genBhLobe(lb) * 10 ** (ipmag[i] / 20)  # lobe magnitudes of the complex exponential
        b = np.arange(round(loc) - 4, round(loc) + 5, dtype='int')
        for m in range(0, 9):
            if b[m] < 0:  # peak lobe crosses DC bin
                Y[-b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] > hN:  # peak lobe croses Nyquist bin
                Y[b[m]] += lmag[m] * np.exp(-1j * ipphase[i])
            elif b[m] == 0 or b[m] == hN:  # peak lobe in the limits of the spectrum
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i]) + lmag[m] * np.exp(-1j * ipphase[i])
            else:  # peak lobe in positive freq. range
                Y[b[m]] += lmag[m] * np.exp(1j * ipphase[i])
        Y[hN + 1:] = Y[hN - 1:0:-1].conjugate()  # fill the negative part of the spectrum
    return Y


def sinewaveSynth(freqs, amp, H, fs):
    """
	Synthesis of one sinusoid with time-varying frequency
	freqs, amps: array of frequencies and amplitudes of sinusoids
	H: hop size, fs: sampling rate
	returns y: output array sound
	"""

    t = np.arange(H) / float(fs)  # time array
    lastphase = 0  # initialize synthesis phase
    lastfreq = freqs[0]  # initialize synthesis frequency
    y = np.array([])  # initialize output array
    for l in range(freqs.size):  # iterate over all frames
        if (lastfreq == 0) & (freqs[l] == 0):  # if 0 freq add zeros
            A = np.zeros(H)
            freq = np.zeros(H)
        elif (lastfreq == 0) & (freqs[l] > 0):  # if starting freq ramp up the amplitude
            A = np.arange(0, amp, amp / H)
            freq = np.ones(H) * freqs[l]
        elif (lastfreq > 0) & (freqs[l] > 0):  # if freqs in boundaries use both
            A = np.ones(H) * amp
            if (lastfreq == freqs[l]):
                freq = np.ones(H) * lastfreq
            else:
                freq = np.arange(lastfreq, freqs[l], (freqs[l] - lastfreq) / H)
        elif (lastfreq > 0) & (freqs[l] == 0):  # if ending freq ramp down the amplitude
            A = np.arange(amp, 0, -amp / H)
            freq = np.ones(H) * lastfreq
        phase = 2 * np.pi * freq * t + lastphase  # generate phase values
        yh = A * np.cos(phase)  # compute sine for one frame
        lastfreq = freqs[l]  # save frequency for phase propagation
        lastphase = np.remainder(phase[H - 1], 2 * np.pi)  # save phase to be use for next frame
        y = np.append(y, yh)  # append frame to previous one
    return y


def cleaningTrack(track, minTrackLength=3):
    """
	Delete fragments of one single track smaller than minTrackLength
	track: array of values; minTrackLength: minimum duration of tracks in number of frames
	returns cleanTrack: array of clean values
	"""

    nFrames = track.size  # number of frames
    cleanTrack = np.copy(track)  # copy array
    trackBegs = np.nonzero((track[:nFrames - 1] <= 0)  # beginning of track contours
                           & (track[1:] > 0))[0] + 1
    if track[0] > 0:
        trackBegs = np.insert(trackBegs, 0, 0)
    trackEnds = np.nonzero((track[:nFrames - 1] > 0) & (track[1:] <= 0))[0] + 1
    if track[nFrames - 1] > 0:
        trackEnds = np.append(trackEnds, nFrames - 1)
    trackLengths = 1 + trackEnds - trackBegs  # lengths of trach contours
    for i, j in zip(trackBegs, trackLengths):  # delete short track contours
        if j <= minTrackLength:
            cleanTrack[i:i + j] = 0
    return cleanTrack


def f0Twm(pfreq, pmag, ef0max, minf0, maxf0, f0t=0):
    """
	Function that wraps the f0 detection function TWM, selecting the possible f0 candidates
	and calling the function TWM with them
	pfreq, pmag: peak frequencies and magnitudes,
	ef0max: maximum error allowed, minf0, maxf0: minimum  and maximum f0
	f0t: f0 of previous frame if stable
	returns f0: fundamental frequency in Hz
	"""
    if (minf0 < 0):  # raise exception if minf0 is smaller than 0
        raise ValueError("Minimum fundamental frequency (minf0) smaller than 0")

    if (maxf0 >= 10000):  # raise exception if maxf0 is bigger than 10000Hz
        raise ValueError("Maximum fundamental frequency (maxf0) bigger than 10000Hz")

    if (pfreq.size < 3) & (f0t == 0):  # return 0 if less than 3 peaks and not previous f0
        return 0

    f0c = np.argwhere((pfreq > minf0) & (pfreq < maxf0))[:, 0]  # use only peaks within given range
    if (f0c.size == 0):  # return 0 if no peaks within range
        return 0
    f0cf = pfreq[f0c]  # frequencies of peak candidates
    f0cm = pmag[f0c]  # magnitude of peak candidates

    if f0t > 0:  # if stable f0 in previous frame
        shortlist = np.argwhere(np.abs(f0cf - f0t) < f0t / 2.0)[:, 0]  # use only peaks close to it
        maxc = np.argmax(f0cm)
        maxcfd = f0cf[maxc] % f0t
        if maxcfd > f0t / 2:
            maxcfd = f0t - maxcfd
        if (maxc not in shortlist) and (maxcfd > (f0t / 4)):  # or the maximum magnitude peak is not a harmonic
            shortlist = np.append(maxc, shortlist)
        f0cf = f0cf[shortlist]  # frequencies of candidates

    if (f0cf.size == 0):  # return 0 if no peak candidates
        return 0

    f0, f0error = UF_C.twm(pfreq, pmag, f0cf)  # call the TWM function with peak candidates, cython version
    #	f0, f0error = TWM_p(pfreq, pmag, f0cf)        # call the TWM function with peak candidates, python version

    if (f0 > 0) and (f0error < ef0max):  # accept and return f0 if below max error allowed
        return f0
    else:
        return 0


def TWM_p(pfreq, pmag, f0c):
    """
	Two-way mismatch algorithm for f0 detection (by Beauchamp&Maher)
	[better to use the C version of this function: UF_C.twm]
	pfreq, pmag: peak frequencies in Hz and magnitudes,
	f0c: frequencies of f0 candidates
	returns f0, f0Error: fundamental frequency detected and its error
	"""

    p = 0.5  # weighting by frequency value
    q = 1.4  # weighting related to magnitude of peaks
    r = 0.5  # scaling related to magnitude of peaks
    rho = 0.33  # weighting of MP error
    Amax = max(pmag)  # maximum peak magnitude
    maxnpeaks = 10  # maximum number of peaks used
    harmonic = np.matrix(f0c)
    ErrorPM = np.zeros(harmonic.size)  # initialize PM errors
    MaxNPM = min(maxnpeaks, pfreq.size)
    for i in range(0, MaxNPM):  # predicted to measured mismatch error
        difmatrixPM = harmonic.T * np.ones(pfreq.size)
        difmatrixPM = abs(difmatrixPM - np.ones((harmonic.size, 1)) * pfreq)
        FreqDistance = np.amin(difmatrixPM, axis=1)  # minimum along rows
        peakloc = np.argmin(difmatrixPM, axis=1)
        Ponddif = np.array(FreqDistance) * (np.array(harmonic.T) ** (-p))
        PeakMag = pmag[peakloc]
        MagFactor = 10 ** ((PeakMag - Amax) / 20)
        ErrorPM = ErrorPM + (Ponddif + MagFactor * (q * Ponddif - r)).T
        harmonic = harmonic + f0c

    ErrorMP = np.zeros(harmonic.size)  # initialize MP errors
    MaxNMP = min(maxnpeaks, pfreq.size)
    for i in range(0, f0c.size):  # measured to predicted mismatch error
        nharm = np.round(pfreq[:MaxNMP] / f0c[i])
        nharm = (nharm >= 1) * nharm + (nharm < 1)
        FreqDistance = abs(pfreq[:MaxNMP] - nharm * f0c[i])
        Ponddif = FreqDistance * (pfreq[:MaxNMP] ** (-p))
        PeakMag = pmag[:MaxNMP]
        MagFactor = 10 ** ((PeakMag - Amax) / 20)
        ErrorMP[i] = sum(MagFactor * (Ponddif + MagFactor * (q * Ponddif - r)))

    Error = (ErrorPM[0] / MaxNPM) + (rho * ErrorMP / MaxNMP)  # total error
    f0index = np.argmin(Error)  # get the smallest error
    f0 = f0c[f0index]  # f0 with the smallest error

    return f0, Error[f0index]


def sineSubtraction(x, N, H, sfreq, smag, sphase, fs):
    """
	Subtract sinusoids from a sound
	x: input sound, N: fft-size, H: hop-size
	sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
	returns xr: residual sound
	"""

    hN = N // 2  # half of fft size
    x = np.append(np.zeros(hN), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hN))  # add zeros at the end to analyze last sample
    bh = blackmanharris(N)  # blackman harris window
    w = bh / sum(bh)  # normalize window
    sw = np.zeros(N)  # initialize synthesis window
    sw[hN - H:hN + H] = triang(2 * H) / w[hN - H:hN + H]  # synthesis window
    L = sfreq.shape[0]  # number of frames, this works if no sines
    xr = np.zeros(x.size)  # initialize output array
    pin = 0
    for l in range(L):
        xw = x[pin:pin + N] * w  # window the input sound
        X = fft(fftshift(xw))  # compute FFT
        Yh = UF_C.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)  # generate spec sines, cython version
        #		Yh = genSpecSines_p(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N, fs)   # generate spec sines, python version
        Xr = X - Yh  # subtract sines from original spectrum
        xrw = np.real(fftshift(ifft(Xr)))  # inverse FFT
        xr[pin:pin + N] += xrw * sw  # overlap-add
        pin += H  # advance sound pointer
    xr = np.delete(xr, range(hN))  # delete half of first window which was added in stftAnal
    xr = np.delete(xr, range(xr.size - hN, xr.size))  # delete half of last window which was added in stftAnal
    return xr


def stochasticResidualAnal(x, N, H, sfreq, smag, sphase, fs, stocf):
    """
	Subtract sinusoids from a sound and approximate the residual with an envelope
	x: input sound, N: fft size, H: hop-size
	sfreq, smag, sphase: sinusoidal frequencies, magnitudes and phases
	fs: sampling rate; stocf: stochastic factor, used in the approximation
	returns stocEnv: stochastic approximation of residual
	"""

    hN = N // 2  # half of fft size
    x = np.append(np.zeros(hN), x)  # add zeros at beginning to center first window at sample 0
    x = np.append(x, np.zeros(hN))  # add zeros at the end to analyze last sample
    bh = blackmanharris(N)  # synthesis window
    w = bh / sum(bh)  # normalize synthesis window
    L = sfreq.shape[0]  # number of frames, this works if no sines
    pin = 0
    for l in range(L):
        xw = x[pin:pin + N] * w  # window the input sound
        X = fft(fftshift(xw))  # compute FFT
        Yh = UF_C.genSpecSines(N * sfreq[l, :] / fs, smag[l, :], sphase[l, :], N)  # generate spec sines, cython version
        #		Yh = genSpecSines_p(N*sfreq[l,:]/fs, smag[l,:], sphase[l,:], N, fs)   # generate spec sines, python version
        Xr = X - Yh  # subtract sines from original spectrum
        mXr = 20 * np.log10(abs(Xr[:hN]))  # magnitude spectrum of residual
        mXrenv = resample(np.maximum(-200, mXr), mXr.size * stocf)  # decimate the mag spectrum
        if l == 0:  # if first frame
            stocEnv = np.array([mXrenv])
        else:  # rest of frames
            stocEnv = np.vstack((stocEnv, np.array([mXrenv])))
        pin += H  # advance sound pointer
    return stocEnv
